use super::*;
use crate::{
    agglomeration,
    analysis::NetworkId,
    learning::{
        features::Observation,
        interface::{
            self, FluentStatus, InterfaceCheckpoint, InterfaceInput, InterfaceModel,
            InterfaceSource, InterfaceTrainOptions,
        },
        latent::{LatentModel, TransitionExample},
        latent_training::LatentCheckpoint,
        model::{MergeModel, ModelConfig, Supervision},
        rollout::RolloutAuditor,
        training::{Adam, Checkpoint, TrainOptions},
    },
    store::{DatasetStore, RecordKind, Split, group_split},
    training::sha256,
};
use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    io::{BufWriter, Write},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    #[serde(default)]
    pub wl: bool,
    pub hidden: usize,
    pub layers: usize,
    pub seed: u64,
    pub split_seed: u64,
    pub max_records: usize,
}
impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            wl: true,
            hidden: 32,
            layers: 3,
            seed: 42,
            split_seed: 42,
            max_records: 100_000,
        }
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SplitCounts {
    pub records: usize,
    pub merge_records: usize,
    pub preferences: usize,
    pub value_targets: usize,
    pub interfaces: usize,
    pub interface_classes: [usize; 3],
    pub transitions: usize,
    pub groups: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportReport {
    pub schema_version: u32,
    pub package: String,
    pub snapshot_id: String,
    pub options: ExportOptions,
    pub records_sha256: String,
    pub templates_sha256: [String; 3],
    pub references_sha256: String,
    pub splits: [SplitCounts; 3],
    pub parameters: [usize; 3],
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedInterface {
    pub input: InterfaceInput,
    pub targets: Vec<usize>,
    pub complete_plan: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedTransition {
    pub action: usize,
    pub after: Observation,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedRecord {
    pub id: String,
    pub group: String,
    pub split: Split,
    pub observation: Observation,
    pub labels: Supervision,
    pub interfaces: Vec<PreparedInterface>,
    pub transitions: Vec<PreparedTransition>,
}

pub fn export_data(run_dir: &std::path::Path, options: &ExportOptions) -> Result<ExportReport> {
    if options.max_records == 0 {
        bail!("max_records must be positive");
    }
    let store = DatasetStore::open(&run_dir.join("data"))?;
    let snapshot = store.snapshot()?;
    let snapshot_id = snapshot.fingerprint()?;
    let root = run_dir.join("prepared");
    fs::create_dir_all(&root)?;
    let _lock = super::lock(&root.join("export.lock"))?;
    let package = sha256(&serde_json::to_vec(&(
        "portable-training-v1",
        &snapshot_id,
        options,
    ))?);
    let directory = root.join(&package);
    fs::create_dir_all(&directory)?;
    if directory.join("manifest.json").exists() {
        let report: ExportReport = read(&directory.join("manifest.json"))?;
        if report.schema_version != 1
            || report.package != package
            || report.snapshot_id != snapshot_id
            || serde_json::to_value(&report.options)? != serde_json::to_value(options)?
        {
            bail!("cached export manifest does not match the selected snapshot/options");
        }
        for (name, expected) in [
            "records.jsonl",
            "references.json",
            "merge-template.json",
            "interface-template.json",
            "latent-template.json",
        ]
        .into_iter()
        .zip([
            &report.records_sha256,
            &report.references_sha256,
            &report.templates_sha256[0],
            &report.templates_sha256[1],
            &report.templates_sha256[2],
        ]) {
            if sha256(&fs::read(directory.join(name))?) != *expected {
                bail!("prepared package checksum mismatch: {name}");
            }
        }
        save(&root.join("current.json"), &report)?;
        return Ok(report);
    }
    let config = ModelConfig {
        hidden: options.hidden,
        layers: options.layers,
    };
    let merge = if options.wl {
        MergeModel::new_wl(config, options.seed)?
    } else {
        MergeModel::new(config, options.seed)?
    };
    let interface = InterfaceModel::new(options.hidden, options.seed)?;
    let latent = LatentModel::new(&merge, options.seed)?;
    let mut counts: [SplitCounts; 3] = Default::default();
    let mut all_groups: [BTreeSet<String>; 3] = Default::default();
    let mut ids: [Vec<String>; 3] = Default::default();
    let mut groups: [BTreeSet<String>; 3] = Default::default();
    let mut latent_ids: [Vec<String>; 3] = Default::default();
    let mut latent_groups: [BTreeSet<String>; 3] = Default::default();
    let mut interface_sources = Vec::new();
    let mut references = Vec::<serde_json::Value>::new();
    let path = directory.join("records.jsonl");
    let temporary = directory.join("records.jsonl.tmp");
    let mut output = BufWriter::new(fs::File::create(&temporary)?);
    let mut auditor = RolloutAuditor::new(&store, &snapshot);
    for entry in snapshot
        .entries
        .iter()
        .filter(|e| e.metadata.kind == RecordKind::Rollout)
        .take(options.max_records)
    {
        let record = auditor.audit(entry)?;
        let labels = record.supervision(1.0)?;
        let split = group_split(&record.group_id, options.split_seed);
        let index = match split {
            Split::Train => 0,
            Split::Validation => 1,
            Split::Test => 2,
        };
        let interfaces = interface::examples_from_rollout(auditor.model(&record)?, &record)?
            .into_iter()
            .map(|e| PreparedInterface {
                input: e.input,
                targets: e
                    .targets
                    .iter()
                    .map(|s| match s {
                        FluentStatus::Unknown => 0,
                        FluentStatus::False => 1,
                        FluentStatus::True => 2,
                    })
                    .collect(),
                complete_plan: e.scope == interface::InterfaceEvidence::CompletePlan,
            })
            .collect::<Vec<_>>();
        let (analysis, graph) = auditor.structure(&record)?;
        let state = agglomeration::observe(
            analysis,
            graph,
            NetworkId::Initial,
            &record.clusters,
            &record.options.agglomeration,
        )?;
        // Audited legacy records retain their content identity. Enrich the portable
        // copy from the original AST, rather than trusting a cached graph.
        let mut observation = record.observation.clone();
        observation.hilg = Some(super::super::hilg::InstanceGraph::build(
            analysis,
            graph,
            &state,
            &observation,
        )?);
        let candidates = agglomeration::legal_candidates(
            graph,
            NetworkId::Initial,
            &state,
            &record.options.agglomeration,
        )?;
        let mut transitions = Vec::new();
        for (action, input) in record.observation.actions.iter().enumerate() {
            let Some((a, b)) = input.pair else {
                continue;
            };
            let candidate = candidates
                .iter()
                .find(|c| {
                    c.left == record.observation.nodes[a].id
                        && c.right == record.observation.nodes[b].id
                })
                .context("audited transition lacks a legal merge")?;
            let clusters = agglomeration::merged_clusters(&state.clusters, candidate)?;
            let child = agglomeration::observe(
                analysis,
                graph,
                NetworkId::Initial,
                &clusters,
                &record.options.agglomeration,
            )?;
            let (mut after, _) =
                Observation::build(analysis, graph, &child, &record.options.agglomeration)?;
            after.context.clone_from(&record.observation.context);
            transitions.push(PreparedTransition { action, after });
        }
        counts[index].records += 1;
        all_groups[index].insert(record.group_id.clone());
        if !labels.preferences.is_empty() || !labels.values.is_empty() {
            counts[index].merge_records += 1;
            ids[index].push(entry.id.clone());
            groups[index].insert(record.group_id.clone());
        }
        counts[index].preferences += labels.preferences.len();
        counts[index].value_targets += labels.values.len();
        counts[index].interfaces += interfaces.len();
        for example in &interfaces {
            for &target in &example.targets {
                counts[index].interface_classes[target] += 1;
            }
        }
        if !interfaces.is_empty() {
            interface_sources.push(InterfaceSource {
                record: entry.id.clone(),
                group: record.group_id.clone(),
                split,
            });
        }
        counts[index].transitions += transitions.len();
        if !transitions.is_empty() {
            latent_ids[index].push(entry.id.clone());
            latent_groups[index].insert(record.group_id.clone());
        }
        let prepared = PreparedRecord {
            id: entry.id.clone(),
            group: record.group_id,
            split,
            observation,
            labels,
            interfaces,
            transitions,
        };
        if references.len() < 2
            || (!prepared.interfaces.is_empty()
                && !references
                    .iter()
                    .any(|r| r.get("interface_loss_gradient").is_some()))
            || (!prepared.transitions.is_empty()
                && !references
                    .iter()
                    .any(|r| r.get("latent_loss_gradient").is_some()))
        {
            let mut reference = serde_json::json!({"record":prepared,"prediction":merge.predict(&prepared.observation)?});
            reference["merge_loss_gradient"] = serde_json::to_value(merge.loss_gradient(
                &prepared.observation,
                &prepared.labels,
                1.0,
            )?)?;
            if let Some(example) = prepared.interfaces.first() {
                let targets: Vec<_> = example
                    .targets
                    .iter()
                    .map(|&c| {
                        [
                            FluentStatus::Unknown,
                            FluentStatus::False,
                            FluentStatus::True,
                        ][c]
                    })
                    .collect();
                reference["interface_prediction"] =
                    serde_json::to_value(interface.predict(&example.input)?)?;
                reference["interface_loss_gradient"] = serde_json::to_value(
                    interface.loss_gradient(&example.input, &targets, &[0.25, 1.0, 1.0])?,
                )?;
            }
            if let Some(transition) = prepared.transitions.first() {
                let hidden = merge.encode(&prepared.observation)?;
                let example = TransitionExample {
                    inputs: latent.inputs(
                        &prepared.observation,
                        &hidden,
                        transition.action,
                        &transition.after,
                    )?,
                    target: merge.encode(&transition.after)?,
                };
                reference["latent_inputs"] = serde_json::to_value(&example.inputs)?;
                reference["latent_target"] = serde_json::to_value(&example.target)?;
                reference["latent_loss_gradient"] =
                    serde_json::to_value(latent.loss_gradient(&example)?)?;
            }
            references.push(reference);
        }
        serde_json::to_writer(&mut output, &prepared)?;
        output.write_all(b"\n")?;
        if counts.iter().map(|c| c.records).sum::<usize>() % 100 == 0 {
            eprintln!(
                "audited/exported {} rollouts",
                counts.iter().map(|c| c.records).sum::<usize>()
            );
        }
    }
    output.flush()?;
    output.get_ref().sync_all()?;
    drop(output);
    fs::rename(&temporary, &path)?;
    if counts[0].merge_records == 0 || counts[0].interfaces == 0 || counts[0].transitions == 0 {
        bail!(
            "prepared training split needs merge/value labels, verified interfaces and merge transitions; counts: {:?}",
            counts[0]
        );
    }
    let train_options = TrainOptions {
        model: config,
        epochs: 100,
        batch_size: 64,
        learning_rate: 0.001,
        seed: options.seed,
        split_seed: options.split_seed,
        max_records: options.max_records,
        ..Default::default()
    };
    let merge_checkpoint = Checkpoint {
        schema_version: 1,
        optimizer: Adam::new(merge.parameters.len()),
        model: merge.clone(),
        completed_epochs: 0,
        options: train_options.clone(),
        snapshot_id: snapshot_id.clone(),
        training_records: ids[0].clone(),
        validation_records: ids[1].clone(),
        test_records: ids[2].clone(),
        training_groups: groups[0].iter().cloned().collect(),
        validation_groups: groups[1].iter().cloned().collect(),
        test_groups: groups[2].iter().cloned().collect(),
    };
    let interface_checkpoint = InterfaceCheckpoint {
        schema_version: 1,
        optimizer: Adam::new(interface.parameters.len()),
        model: interface,
        completed_epochs: 0,
        options: InterfaceTrainOptions {
            hidden: options.hidden,
            epochs: 100,
            batch_size: 64,
            learning_rate: 0.001,
            seed: options.seed,
            split_seed: options.split_seed,
            max_records: options.max_records,
            ..Default::default()
        },
        snapshot_id: snapshot_id.clone(),
        sources: interface_sources,
    };
    let latent_checkpoint = LatentCheckpoint {
        schema_version: 1,
        encoder: merge,
        optimizer: Adam::new(latent.parameters.len()),
        model: latent,
        options: train_options,
        completed_epochs: 0,
        snapshot_id: snapshot_id.clone(),
        records: latent_ids,
        groups: latent_groups.map(|g| g.into_iter().collect()),
    };
    merge_checkpoint.save(&directory.join("merge-template.json"))?;
    interface_checkpoint.save(&directory.join("interface-template.json"))?;
    latent_checkpoint.save(&directory.join("latent-template.json"))?;
    save(&directory.join("references.json"), &references)?;
    for (i, set) in all_groups.iter().enumerate() {
        counts[i].groups = set.len();
    }
    let report = ExportReport {
        schema_version: 1,
        package,
        snapshot_id,
        options: options.clone(),
        records_sha256: sha256(&fs::read(&path)?),
        templates_sha256: [
            "merge-template.json",
            "interface-template.json",
            "latent-template.json",
        ]
        .map(|name| fs::read(directory.join(name)).map(|b| sha256(&b)))
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()?
        .try_into()
        .unwrap(),
        references_sha256: sha256(&fs::read(directory.join("references.json"))?),
        splits: counts,
        parameters: [
            merge_checkpoint.model.parameters.len(),
            interface_checkpoint.model.parameters.len(),
            latent_checkpoint.model.parameters.len(),
        ],
    };
    save(&directory.join("manifest.json"), &report)?;
    save(&root.join("current.json"), &report)?;
    Ok(report)
}
