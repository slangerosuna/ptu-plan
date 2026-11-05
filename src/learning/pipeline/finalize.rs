use super::*;
use crate::{
    learning::{
        interface::InterfaceCheckpoint, latent_training::LatentCheckpoint, model::Prediction,
        training::Checkpoint,
    },
    training::sha256,
};
use anyhow::bail;
use std::io::{BufRead, BufReader};

#[derive(serde::Deserialize)]
struct Parity {
    record: String,
    prediction: Prediction,
    interface: Option<Vec<[f64; 3]>>,
    latent: Option<Vec<Vec<f64>>>,
}
fn close(actual: &[f64], expected: &[f64]) -> Result<()> {
    if actual.len() != expected.len()
        || actual.iter().zip(expected).any(|(a, b)| {
            !a.is_finite() || !b.is_finite() || (a - b).abs() > 1e-4 * (1.0 + b.abs())
        })
    {
        bail!("GPU/native inference parity mismatch");
    }
    Ok(())
}

pub fn finalize_bundle(run_dir: &Path, model_dir: &Path) -> Result<()> {
    // The training job pins an immutable package. A concurrent preparer may
    // publish a new current.json without changing this job's source dataset.
    let report: serde_json::Value = read(&model_dir.join("training-report.json"))?;
    let package_id = report
        .get("package")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("training report lacks pinned package"))?;
    if package_id.len() != 64
        || !package_id
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        bail!("invalid training package ID");
    }
    let manifest: super::export::ExportReport = read(
        &run_dir
            .join("prepared")
            .join(package_id)
            .join("manifest.json"),
    )?;
    if manifest.package != package_id || manifest.schema_version != 1 {
        bail!("training package manifest mismatch");
    }
    let package = run_dir.join("prepared").join(&manifest.package);
    for (name, expected) in [
        "records.jsonl",
        "references.json",
        "merge-template.json",
        "interface-template.json",
        "latent-template.json",
    ]
    .into_iter()
    .zip([
        &manifest.records_sha256,
        &manifest.references_sha256,
        &manifest.templates_sha256[0],
        &manifest.templates_sha256[1],
        &manifest.templates_sha256[2],
    ]) {
        if sha256(&fs::read(package.join(name))?) != *expected {
            bail!("prepared package checksum mismatch: {name}");
        }
    }
    let merge: Checkpoint = read(&model_dir.join("merge-policy.json"))?;
    let interface: InterfaceCheckpoint = read(&model_dir.join("interface-policy.json"))?;
    let mut latent: LatentCheckpoint = read(&model_dir.join("latent-policy.json"))?;
    if merge.snapshot_id != manifest.snapshot_id
        || interface.snapshot_id != manifest.snapshot_id
        || latent.snapshot_id != manifest.snapshot_id
        || serde_json::to_vec(&merge.model)? != serde_json::to_vec(&latent.encoder)?
    {
        bail!("model bundle snapshot/encoder mismatch");
    }
    let mt: Checkpoint = read(&package.join("merge-template.json"))?;
    let it: InterfaceCheckpoint = read(&package.join("interface-template.json"))?;
    let lt: LatentCheckpoint = read(&package.join("latent-template.json"))?;
    if (
        &merge.training_records,
        &merge.validation_records,
        &merge.test_records,
        &merge.training_groups,
        &merge.validation_groups,
        &merge.test_groups,
    ) != (
        &mt.training_records,
        &mt.validation_records,
        &mt.test_records,
        &mt.training_groups,
        &mt.validation_groups,
        &mt.test_groups,
    ) || serde_json::to_value(&interface.sources)? != serde_json::to_value(&it.sources)?
        || latent.records != lt.records
        || latent.groups != lt.groups
        || [
            merge.completed_epochs,
            interface.completed_epochs,
            latent.completed_epochs,
            merge.optimizer.step,
            interface.optimizer.step,
            latent.optimizer.step,
        ]
        .contains(&0)
        || [
            merge.options.seed,
            interface.options.seed,
            latent.options.seed,
        ]
        .iter()
        .any(|s| *s != manifest.options.seed)
        || [
            merge.options.split_seed,
            interface.options.split_seed,
            latent.options.split_seed,
        ]
        .iter()
        .any(|s| *s != manifest.options.split_seed)
        || merge.model.schema_version != if manifest.options.wl { 2 } else { 1 }
        || merge.model.config.hidden != manifest.options.hidden
        || merge.model.config.layers != manifest.options.layers
        || interface.model.hidden != manifest.options.hidden
    {
        bail!("model bundle training provenance/config mismatch");
    }
    latent.model.encoder_sha256 = crate::training::sha256(&serde_json::to_vec(&merge.model)?);
    merge.validate()?;
    interface.validate()?;
    latent.validate()?;
    let checks: Vec<Parity> = read(&model_dir.join("parity.json"))?;
    let references: Vec<serde_json::Value> = read(&package.join("references.json"))?;
    let expected: std::collections::BTreeSet<_> = references
        .iter()
        .map(|r| {
            r.get("record")
                .and_then(|r| r.get("id"))
                .and_then(|id| id.as_str())
                .map(String::from)
                .ok_or_else(|| anyhow::anyhow!("reference lacks record ID"))
        })
        .collect::<Result<_>>()?;
    let mut found = std::collections::BTreeSet::new();
    for line in BufReader::new(fs::File::open(package.join("records.jsonl"))?).lines() {
        let record: super::export::PreparedRecord = serde_json::from_str(&line?)?;
        let Some(check) = checks.iter().find(|c| c.record == record.id) else {
            continue;
        };
        found.insert(record.id);
        let prediction = merge.model.predict(&record.observation)?;
        close(&prediction.logits, &check.prediction.logits)?;
        close(&prediction.log_effort, &check.prediction.log_effort)?;
        if let Some(example) = record.interfaces.first() {
            let prediction = interface.model.predict(&example.input)?;
            let expected = check
                .interface
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing interface parity"))?;
            close(
                &prediction.into_iter().flatten().collect::<Vec<_>>(),
                &expected.iter().flatten().copied().collect::<Vec<_>>(),
            )?;
        }
        if let Some(transition) = record.transitions.first() {
            let hidden = merge.model.encode(&record.observation)?;
            let prediction = latent.model.transition(
                &record.observation,
                &hidden,
                transition.action,
                &transition.after,
            )?;
            let expected = check
                .latent
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing latent parity"))?;
            if prediction.len() != expected.len() {
                bail!("latent parity row mismatch");
            }
            for (a, b) in prediction.iter().zip(expected) {
                close(a, b)?;
            }
        }
    }
    if checks.is_empty()
        || found.len() != checks.len()
        || found != expected
        || expected.len() != references.len()
    {
        bail!("missing or duplicate model parity records");
    }
    merge.save(&model_dir.join("merge-policy.json"))?;
    interface.save(&model_dir.join("interface-policy.json"))?;
    latent.save(&model_dir.join("latent-policy.json"))?;
    let mut files = std::collections::BTreeMap::new();
    for name in [
        "merge-policy.json",
        "interface-policy.json",
        "latent-policy.json",
        "training-report.json",
        "parity.json",
    ] {
        files.insert(name, sha256(&fs::read(model_dir.join(name))?));
    }
    save(
        &model_dir.join("bundle.json"),
        &serde_json::json!({"schema_version":1,"package":manifest.package,"snapshot_id":manifest.snapshot_id,"encoder_sha256":latent.model.encoder_sha256,"files":files,"parity_records":found.len()}),
    )?;
    Ok(())
}
