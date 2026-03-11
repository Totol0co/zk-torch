#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_imports)]
pub mod basic_block;
pub mod graph;
pub mod layer;
pub mod onnx;
pub mod ptau;
#[cfg(test)]
pub mod tests;
pub mod util;
pub mod python; 

use std::env;
use std::{fs, path::Path};
use std::io::Read;
use once_cell::sync::OnceCell;


pub static CONFIG: OnceCell<util::Config> = OnceCell::new();
pub static LAYER_SETUP_DIR: OnceCell<String> = OnceCell::new();

/// Initialize the global CONFIG and derived LAYER_SETUP_DIR once (CLI or Python).
pub fn init_from_yaml(yaml: &str) -> Result<(), String> {
    let cfg: util::Config = serde_yaml::from_str(yaml).map_err(|e| format!("bad yaml: {e}"))?;
    CONFIG.set(cfg).map_err(|_| "CONFIG already initialized".to_string())?;

    let cfg = CONFIG.get().unwrap();
    let dir = format!(
        "layer_setup/{}_{}_{}",
        cfg.sf.scale_factor_log, cfg.sf.cq_range_log, cfg.sf.cq_range_lower_log
    );
    if !(Path::new(&dir).exists() || fs::create_dir_all(&dir).is_ok()) {
        return Err(format!("cannot create {dir}"));
    }
    LAYER_SETUP_DIR.set(dir).map_err(|_| "LAYER_SETUP_DIR already initialized".to_string())?;
    Ok(())
}
