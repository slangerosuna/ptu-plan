//! Single-job preparation and portable GPU training packages.
mod export;
mod finalize;
mod prepare;
use anyhow::Result;
pub use export::{ExportOptions, ExportReport, export_data};
pub use finalize::finalize_bundle;
pub use prepare::{PrepareOptions, PrepareReport, prepare_data};
use serde::{Serialize, de::DeserializeOwned};
use std::{fs, path::Path};
struct RunLock(fs::File);
impl Drop for RunLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}
fn lock(path: &Path) -> Result<RunLock> {
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(path)?;
    file.try_lock()
        .map_err(|e| anyhow::anyhow!("another process owns {}: {e}", path.display()))?;
    Ok(RunLock(file))
}
pub(super) fn save<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    crate::store::atomic_write(path, &serde_json::to_vec_pretty(value)?)
}
pub(super) fn read<T: DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
