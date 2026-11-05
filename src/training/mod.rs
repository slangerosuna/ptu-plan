//! Validated, self-contained training records with explicit label provenance.

pub mod backward;
pub mod panda;

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::analysis::{HtnAnalysis, analyze_program};
use crate::interaction::InteractionGraph;
use crate::parsing::json::Program;
use crate::partition::{MicroPartitioning, SpectralOptions, micropartition_network};
use crate::solver::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Provenance {
    Backward {
        seed: u64,
        attempt: usize,
        template: String,
    },
    PandaOptimal {
        domain_path: PathBuf,
        problem_path: PathBuf,
        evidence: Box<panda::PandaEvidence>,
    },
}

impl Provenance {
    pub fn source(&self) -> &'static str {
        match self {
            Self::Backward { .. } => "backward",
            Self::PandaOptimal { .. } => "panda-optimal",
        }
    }
    pub fn optimal(&self) -> bool {
        matches!(self, Self::PandaOptimal { .. })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryLabel {
    pub after_actions: usize,
    pub requirements: BoundaryState,
    pub remaining_actions: usize,
    /// Feasible upper bounds for synthetic samples. Exact only with external
    /// cost-optimal evidence, for this residual HTN state, not arbitrary states.
    pub exact_remaining_cost: bool,
}

#[derive(Debug, Serialize)]
pub struct TrainingSample {
    pub schema_version: u32,
    #[serde(skip_serializing_if = "GroundingMode::is_full")]
    pub grounding_mode: GroundingMode,
    pub id: String,
    /// Domain-level grouping keeps traces/cuts and synthetic siblings together.
    pub group_id: String,
    pub source: Provenance,
    pub program: Program,
    pub certificate: PlanCertificate,
    pub trace: Vec<ExecutionStep>,
    pub boundaries: Vec<BoundaryLabel>,
    pub action_cost: usize,
    pub optimal: bool,
    pub analysis: HtnAnalysis,
    pub graph: InteractionGraph,
    pub micropartitions: MicroPartitioning,
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

// JSON float parsing and eigensolvers can differ by a few ULPs. Membership,
// integer metadata, keys, and array order must still agree exactly.
pub(crate) fn spectral_cache_matches(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use serde_json::Value;
    match (a, b) {
        (Value::Number(a), Value::Number(b)) if a.is_f64() && b.is_f64() => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| spectral_cache_matches(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, a)| b.get(key).is_some_and(|b| spectral_cache_matches(a, b)))
        }
        _ => a == b,
    }
}

pub fn sample_fingerprint(program: &Program, actions: &[GroundTask]) -> Result<String> {
    let mut canonical = program.clone();
    if let Some(problem) = &mut canonical.problem {
        problem.name.clear();
    }
    Ok(sha256(&serde_json::to_vec(&(&canonical, actions))?))
}

/// Assemble features only after checking the exact symbolic witness.
pub fn make_sample(
    program: Program,
    model: &GroundProblem,
    certificate: PlanCertificate,
    source: Provenance,
) -> Result<TrainingSample> {
    validate_problem_certificate(model, &certificate)?;
    let initial = &model.initial_networks[certificate.initial_network];
    let request = LocalRequest {
        network: initial.network.clone(),
        initial_state: model.initial_state.clone(),
        goal: model.goal.clone(),
    };
    let trace = trace_certificate(model, &request, &certificate)?;
    let boundaries = backward::boundary_labels(model, &trace, &model.goal, source.optimal())?;
    let analysis = analyze_program(&program)?;
    let graph = InteractionGraph::from_analysis(&analysis);
    let micropartitions = micropartition_network(
        &graph,
        crate::analysis::NetworkId::Initial,
        &SpectralOptions::default(),
    )?;
    let group_id = sha256(&serde_json::to_vec(&program.domain)?);
    let id = format!(
        "{}-{}",
        source.source(),
        sample_fingerprint(&program, &certificate.actions)?
    );
    Ok(TrainingSample {
        schema_version: 1,
        grounding_mode: model.grounding_mode,
        id,
        group_id,
        optimal: source.optimal(),
        source,
        program,
        action_cost: certificate.actions.len(),
        certificate,
        trace,
        boundaries,
        analysis,
        graph,
        micropartitions,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub file: String,
    pub sha256: String,
    pub source: String,
    pub group_id: String,
    pub action_cost: usize,
    pub optimal: bool,
}

pub struct DatasetWriter {
    root: PathBuf,
}

impl DatasetWriter {
    pub fn new(root: &Path) -> Result<Self> {
        fs::create_dir_all(root)?;
        Ok(Self {
            root: root.to_path_buf(),
        })
    }

    /// Content-derived names make reruns additive. Existing records are never
    /// overwritten; malformed/incompatible existing data is an error.
    pub fn write(&self, sample: &TrainingSample) -> Result<bool> {
        if sample.id.contains('/') || sample.id.contains('\\') || sample.id.starts_with('.') {
            bail!("unsafe sample ID");
        }
        let path = self.root.join(format!("{}.json", sample.id));
        let bytes = serde_json::to_vec(sample)?;
        if path.exists() {
            let old: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
            if old["id"] != sample.id || old["schema_version"] != 1 {
                bail!("existing record is incompatible: {}", path.display());
            }
            return Ok(false);
        }
        let temporary = self
            .root
            .join(format!(".{}.{}.tmp", sample.id, std::process::id()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        let published = fs::hard_link(&temporary, &path);
        fs::remove_file(&temporary)?;
        match published {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    fn entries(&self) -> Result<Vec<ManifestEntry>> {
        let mut paths: Vec<_> = fs::read_dir(&self.root)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<std::io::Result<_>>()?;
        paths.sort();
        let mut entries = Vec::new();
        for path in paths {
            let name = path.file_name().unwrap().to_string_lossy();
            if !(name.starts_with("backward-") || name.starts_with("panda-optimal-"))
                || path.extension().is_none_or(|x| x != "json")
            {
                continue;
            }
            let bytes = fs::read(&path)?;
            let value: serde_json::Value = serde_json::from_slice(&bytes)
                .with_context(|| format!("reading {}", path.display()))?;
            if value["schema_version"] != 1
                || name != format!("{}.json", value["id"].as_str().unwrap_or(""))
            {
                bail!("invalid training record {}", path.display());
            }
            let source: Provenance = serde_json::from_value(value["source"].clone())?;
            entries.push(ManifestEntry {
                file: name.into_owned(),
                sha256: sha256(&bytes),
                source: source.source().into(),
                group_id: value["group_id"]
                    .as_str()
                    .context("missing group_id")?
                    .into(),
                action_cost: value["action_cost"]
                    .as_u64()
                    .context("missing action cost")? as usize,
                optimal: source.optimal(),
            });
        }
        Ok(entries)
    }

    pub fn manifest(&self) -> Result<Vec<ManifestEntry>> {
        let entries = self.entries()?;
        let manifest = serde_json::to_vec_pretty(&entries)?;
        let temporary = self.root.join("manifest.json.tmp");
        fs::write(&temporary, manifest)?;
        fs::rename(temporary, self.root.join("manifest.json"))?;
        Ok(entries)
    }

    pub fn audit(&self, options: &GroundingOptions) -> Result<usize> {
        let manifest_path = self.root.join("manifest.json");
        if manifest_path.exists() {
            let previous: Vec<ManifestEntry> = serde_json::from_slice(&fs::read(manifest_path)?)?;
            for entry in previous {
                if Path::new(&entry.file)
                    .file_name()
                    .is_none_or(|name| name != entry.file.as_str())
                {
                    bail!("unsafe manifest path");
                }
                if sha256(&fs::read(self.root.join(&entry.file))?) != entry.sha256 {
                    bail!("record checksum mismatch: {}", entry.file);
                }
            }
        }
        let entries = self.entries()?;
        for entry in &entries {
            let value: serde_json::Value =
                serde_json::from_slice(&fs::read(self.root.join(&entry.file))?)?;
            let program: Program = serde_json::from_value(value["program"].clone())?;
            let certificate: PlanCertificate =
                serde_json::from_value(value["certificate"].clone())?;
            let mode = value
                .get("grounding_mode")
                .map(|v| serde_json::from_value(v.clone()))
                .transpose()?
                .unwrap_or_default();
            let model = match mode {
                GroundingMode::Full => ground_program(&program, options)?,
                GroundingMode::InitialReachable => ground_reachable_program(&program, options)?,
            };
            validate_problem_certificate(&model, &certificate)
                .with_context(|| format!("replaying {}", entry.file))?;
            if entry.action_cost != certificate.actions.len() || value["optimal"] != entry.optimal {
                bail!("incorrect labels in {}", entry.file);
            }
            let source: Provenance = serde_json::from_value(value["source"].clone())?;
            if let Provenance::PandaOptimal { evidence, .. } = &source {
                evidence.validate(entry.action_cost)?;
                if panda::primitive_plan(&evidence.converted_plan)? != certificate.actions {
                    bail!("PANDA primitive plan differs from replay witness");
                }
            }
            let initial = &model.initial_networks[certificate.initial_network];
            let request = LocalRequest {
                network: initial.network.clone(),
                initial_state: model.initial_state.clone(),
                goal: model.goal.clone(),
            };
            let trace = trace_certificate(&model, &request, &certificate)?;
            let boundaries =
                backward::boundary_labels(&model, &trace, &model.goal, source.optimal())?;
            if value["trace"] != serde_json::to_value(&trace)?
                || value["boundaries"] != serde_json::to_value(boundaries)?
            {
                bail!(
                    "trace or boundary labels disagree with replay in {}",
                    entry.file
                );
            }
            if entry.group_id != sha256(&serde_json::to_vec(&program.domain)?) {
                bail!("incorrect domain group ID");
            }
            let analysis = analyze_program(&program)?;
            let graph = InteractionGraph::from_analysis(&analysis);
            if value["analysis"] != serde_json::to_value(analysis)?
                || value["graph"] != serde_json::to_value(&graph)?
            {
                bail!(
                    "cached analysis/graph differs from program in {}",
                    entry.file
                );
            }
            let micropartitions = micropartition_network(
                &graph,
                crate::analysis::NetworkId::Initial,
                &SpectralOptions::default(),
            )?;
            if !spectral_cache_matches(
                &value["micropartitions"],
                &serde_json::to_value(micropartitions)?,
            ) {
                bail!(
                    "cached micropartitions differ from program in {}",
                    entry.file
                );
            }
            let expected_id = format!(
                "{}-{}",
                entry.source,
                sample_fingerprint(&program, &certificate.actions)?
            );
            if value["id"] != expected_id {
                bail!("content ID mismatch in {}", entry.file);
            }
        }
        // Rebuild an interrupted worker's missing manifest only after every
        // discovered record passes replay and label validation.
        self.manifest()?;
        Ok(entries.len())
    }
}
