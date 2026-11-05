use ptu_plan::store::*;
use ptu_plan::training::sha256;
use serde_json::json;
use std::fs::{self, OpenOptions};

fn metadata() -> Metadata {
    Metadata {
        kind: RecordKind::Problem,
        group_id: Some(sha256(b"domain")),
        source_id: None,
    }
}

#[test]
fn immutable_objects_deduplicate_and_readers_keep_stable_snapshots() {
    let root = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(root.path()).unwrap();
    let first = store.put(metadata(), &json!({"value":1})).unwrap();
    let snapshot = store.snapshot().unwrap();
    assert_eq!(first, store.put(metadata(), &json!({"value":1})).unwrap());
    let second = store.put(metadata(), &json!({"value":2})).unwrap();
    assert_ne!(first.id, second.id);
    assert_eq!(snapshot.entries.len(), 1);
    assert_eq!(store.snapshot().unwrap().entries.len(), 2);
    assert_eq!(
        store
            .load::<serde_json::Value>(&snapshot.entries[0])
            .unwrap(),
        json!({"value":1})
    );
    assert_ne!(
        snapshot.fingerprint().unwrap(),
        store.snapshot().unwrap().fingerprint().unwrap()
    );
}

#[test]
fn recovery_finds_unindexed_objects_and_refuses_corruption() {
    let root = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(root.path()).unwrap();
    let entry = store.put(metadata(), &json!({"safe":true})).unwrap();
    fs::remove_file(root.path().join("index.json")).unwrap();
    assert!(store.snapshot().unwrap().entries.is_empty());
    assert_eq!(
        store.repair().unwrap().entries.as_slice(),
        std::slice::from_ref(&entry)
    );
    let path = root
        .path()
        .join("objects")
        .join(&entry.id[..2])
        .join(format!("{}.json", entry.id));
    fs::write(path, b"{}").unwrap();
    assert!(
        store
            .load::<serde_json::Value>(&entry)
            .unwrap_err()
            .to_string()
            .contains("checksum")
    );
    assert!(store.repair().is_err());
}

#[test]
fn writer_lock_prevents_competing_commits_without_blocking_snapshot_reads() {
    let root = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(root.path()).unwrap();
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(root.path().join(".store.lock"))
        .unwrap();
    lock.try_lock().unwrap();
    assert!(
        store
            .put(metadata(), &json!({}))
            .unwrap_err()
            .to_string()
            .contains("writer")
    );
    assert!(store.snapshot().unwrap().entries.is_empty());
    drop(lock);
    assert!(store.put(metadata(), &json!({})).is_ok());
}

#[test]
fn metadata_and_domain_splits_are_validated() {
    let root = tempfile::tempdir().unwrap();
    let store = DatasetStore::open(root.path()).unwrap();
    let mut bad = metadata();
    bad.group_id = Some("../domain".into());
    assert!(store.put(bad, &json!({})).is_err());
    let mut entry = store.put(metadata(), &json!({})).unwrap();
    entry.metadata.source_id = Some("changed".into());
    assert!(store.load::<serde_json::Value>(&entry).is_err());
    let splits: std::collections::BTreeSet<_> = (0..100)
        .map(|i| format!("{:?}", group_split(&sha256(i.to_string().as_bytes()), 42)))
        .collect();
    assert_eq!(splits.len(), 3);
    for group in ["domain-a", "domain-b"] {
        assert_eq!(group_split(group, 42), group_split(group, 42));
    }
}
