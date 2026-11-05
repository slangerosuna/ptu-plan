//! Immutable, content-addressed experiment records with atomic snapshot indexes.
//! The legacy solution dataset is read through a checksum-verified snapshot.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::parsing::json::Program;
use crate::training::{ManifestEntry, sha256};

static TEMPORARY: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum RecordKind {
    Problem,
    Rollout,
    Model,
    TrainingReport,
    CollectionFailure,
    InterfaceModel,
    InterfaceTrainingReport,
    LatentModel,
    LatentTrainingReport,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct Metadata {
    pub kind: RecordKind,
    pub group_id: Option<String>,
    pub source_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct Entry {
    pub id: String,
    pub metadata: Metadata,
    pub bytes: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct Envelope {
    schema_version: u32,
    metadata: Metadata,
    payload: serde_json::Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Snapshot {
    pub schema_version: u32,
    pub entries: Vec<Entry>,
}

impl Snapshot {
    pub fn fingerprint(&self) -> Result<String> {
        Ok(sha256(&serde_json::to_vec(self)?))
    }
}

pub struct DatasetStore {
    root: PathBuf,
}

struct StoreLock(File);
impl Drop for StoreLock {
    fn drop(&mut self) {
        // Closing alone can leave a lock held by an inherited/duplicated file
        // description until a concurrent subprocess reaches exec or exits.
        let _ = self.0.unlock();
    }
}

fn hash_name(name: &str) -> bool {
    name.len() == 64
        && name
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("missing parent directory")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".write-{}-{}.tmp",
        std::process::id(),
        TEMPORARY.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let mut output = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        output.write_all(bytes)?;
        output.sync_all()?;
        drop(output);
        fs::rename(&temporary, path)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if temporary.exists() {
        let _ = fs::remove_file(temporary);
    }
    result
}

impl DatasetStore {
    pub fn open(root: &Path) -> Result<Self> {
        if root.is_symlink() {
            bail!("store root must not be a symlink");
        }
        fs::create_dir_all(root.join("objects"))?;
        let root = fs::canonicalize(root)?;
        File::open(root.join("objects"))?.sync_all()?;
        File::open(&root)?.sync_all()?;
        if let Some(parent) = root.parent() {
            File::open(parent)?.sync_all()?;
        }
        Ok(Self { root })
    }

    fn lock(&self) -> Result<StoreLock> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.root.join(".store.lock"))?;
        file.try_lock()
            .context("dataset store has another active writer")?;
        Ok(StoreLock(file))
    }

    fn path(&self, id: &str) -> Result<PathBuf> {
        if !hash_name(id) {
            bail!("invalid store record ID");
        }
        Ok(self
            .root
            .join("objects")
            .join(&id[..2])
            .join(format!("{id}.json")))
    }

    pub fn snapshot(&self) -> Result<Snapshot> {
        let path = self.root.join("index.json");
        if !path.exists() {
            return Ok(Snapshot {
                schema_version: 1,
                entries: Vec::new(),
            });
        }
        let snapshot: Snapshot = serde_json::from_slice(&fs::read(path)?)?;
        if snapshot.schema_version != 1 {
            bail!("unsupported store schema");
        }
        let mut seen = BTreeSet::new();
        for entry in &snapshot.entries {
            if !hash_name(&entry.id) || !seen.insert(&entry.id) {
                bail!("invalid or duplicate index entry");
            }
        }
        Ok(snapshot)
    }

    fn envelope(&self, entry: &Entry) -> Result<Envelope> {
        let path = self.path(&entry.id)?;
        if path.is_symlink() {
            bail!("store object must not be a symlink");
        }
        let bytes =
            fs::read(&path).with_context(|| format!("reading store object {}", entry.id))?;
        if bytes.len() != entry.bytes || sha256(&bytes) != entry.id {
            bail!("store checksum mismatch: {}", entry.id);
        }
        let envelope: Envelope = serde_json::from_slice(&bytes)?;
        if envelope.schema_version != 1 || envelope.metadata != entry.metadata {
            bail!("store object metadata mismatch");
        }
        Ok(envelope)
    }

    pub fn load<T: DeserializeOwned>(&self, entry: &Entry) -> Result<T> {
        Ok(serde_json::from_value(self.envelope(entry)?.payload)?)
    }

    pub fn put<T: Serialize>(&self, metadata: Metadata, payload: &T) -> Result<Entry> {
        if metadata.group_id.as_ref().is_some_and(|id| !hash_name(id)) {
            bail!("group ID must be a SHA-256 digest");
        }
        let envelope = Envelope {
            schema_version: 1,
            metadata: metadata.clone(),
            payload: serde_json::to_value(payload)?,
        };
        let bytes = serde_json::to_vec(&envelope)?;
        let entry = Entry {
            id: sha256(&bytes),
            metadata,
            bytes: bytes.len(),
        };
        let _lock = self.lock()?;
        let path = self.path(&entry.id)?;
        if path.exists() {
            self.envelope(&entry)?;
        } else {
            // Writers hold the directory lock; existing immutable objects are
            // checked, never replaced. The index commits only after the object.
            atomic_write(&path, &bytes)?;
            File::open(self.root.join("objects"))?.sync_all()?;
        }
        let mut snapshot = self.snapshot()?;
        if let Some(old) = snapshot.entries.iter().find(|old| old.id == entry.id) {
            if *old != entry {
                bail!("conflicting store index entry");
            }
        } else {
            snapshot.entries.push(entry.clone());
            snapshot.entries.sort_by(|a, b| a.id.cmp(&b.id));
            atomic_write(
                &self.root.join("index.json"),
                &serde_json::to_vec_pretty(&snapshot)?,
            )?;
        }
        Ok(entry)
    }

    /// Recover objects committed before an interrupted index update. Every
    /// referenced object must remain intact; corruption is never dropped.
    pub fn repair(&self) -> Result<Snapshot> {
        let _lock = self.lock()?;
        let old = self.snapshot()?;
        for entry in &old.entries {
            self.envelope(entry)?;
        }
        let mut entries = BTreeMap::new();
        for directory in fs::read_dir(self.root.join("objects"))? {
            let directory = directory?.path();
            if directory.is_symlink() || !directory.is_dir() {
                bail!("invalid object shard");
            }
            for object in fs::read_dir(&directory)? {
                let path = object?.path();
                if path.extension().is_none_or(|s| s != "json") {
                    continue;
                }
                let id = path
                    .file_stem()
                    .context("missing object name")?
                    .to_string_lossy()
                    .to_string();
                if self.path(&id)? != path {
                    bail!("object in wrong shard");
                }
                let envelope: Envelope = serde_json::from_slice(&fs::read(&path)?)?;
                let entry = Entry {
                    id: id.clone(),
                    metadata: envelope.metadata,
                    bytes: fs::metadata(&path)?.len() as usize,
                };
                self.envelope(&entry)?;
                entries.insert(id, entry);
            }
        }
        let snapshot = Snapshot {
            schema_version: 1,
            entries: entries.into_values().collect(),
        };
        atomic_write(
            &self.root.join("index.json"),
            &serde_json::to_vec_pretty(&snapshot)?,
        )?;
        Ok(snapshot)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Split {
    Train,
    Validation,
    Test,
}

/// Every trace, counterfactual, and synthetic sibling of a domain shares a split.
pub fn group_split(group: &str, seed: u64) -> Split {
    let hash = sha256(format!("ptu-plan-split-v1:{seed}:{group}").as_bytes());
    let bucket = u64::from_str_radix(&hash[..16], 16).unwrap() % 100;
    match bucket {
        0..=79 => Split::Train,
        80..=89 => Split::Validation,
        _ => Split::Test,
    }
}

pub struct SolutionSnapshot {
    root: PathBuf,
    pub entries: Vec<ManifestEntry>,
}

impl SolutionSnapshot {
    pub fn open(root: &Path) -> Result<Self> {
        let entries: Vec<ManifestEntry> =
            serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
        let mut seen = BTreeSet::new();
        for entry in &entries {
            if Path::new(&entry.file)
                .file_name()
                .is_none_or(|n| n != entry.file.as_str())
                || !seen.insert(&entry.file)
            {
                bail!("unsafe or duplicate solution manifest entry");
            }
        }
        Ok(Self {
            root: root.to_path_buf(),
            entries,
        })
    }

    pub fn program(&self, entry: &ManifestEntry) -> Result<Program> {
        let path = self.root.join(&entry.file);
        if path.is_symlink() {
            bail!("solution record must not be a symlink");
        }
        let bytes = fs::read(path)?;
        if sha256(&bytes) != entry.sha256 {
            bail!("solution checksum mismatch");
        }
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        let program: Program = serde_json::from_value(value["program"].clone())?;
        if value["schema_version"] != 1
            || value["group_id"] != entry.group_id
            || sha256(&serde_json::to_vec(&program.domain)?) != entry.group_id
        {
            bail!("solution metadata mismatch");
        }
        Ok(program)
    }
}

#[cfg(test)]
mod lock_tests {
    use super::*;
    #[test]
    fn releasing_writer_guard_unlocks_even_if_another_handle_survives() {
        let directory = tempfile::tempdir().unwrap();
        let store = DatasetStore::open(directory.path()).unwrap();
        let guard = store.lock().unwrap();
        let inherited = guard.0.try_clone().unwrap();
        assert!(store.lock().is_err());
        drop(guard);
        let next = store
            .lock()
            .expect("a surviving duplicate must not prolong the finished writer's lock");
        drop(inherited);
        assert!(
            store.lock().is_err(),
            "closing an old handle must not release a new writer's lock"
        );
        drop(next);
        assert!(store.lock().is_ok());
    }
}
