//! Python bindings for ZK Torch 
//!
//! Python API:
//!   - setup(config_path: str)
//!   - prove(config_path: str)
//!   - verify(config_path: str)
//!


use pyo3::prelude::*;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use ndarray::ArrayD;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_poly::univariate::DensePolynomial;
use ark_serialize::CanonicalDeserialize;

use crate::{basic_block::{Data, SRS},onnx,ptau,util,};
use crate::util::{prover, verifier};
use crate::{init_from_yaml, CONFIG};
use crate::util::Config;
use plonky2::{timed, util::timing::TimingTree};

/// Ensure CONFIG is initialized from the provided YAML path exactly once
fn ensure_cfg(config_path: &str) -> &'static Config {
    if CONFIG.get().is_none() {
        let yaml = std::fs::read_to_string(config_path)
            .unwrap_or_else(|e| panic!("Cannot read config file '{}': {e}", config_path));
        // It is okay if CONFIG was already initialized elsewhere; we only attempt when None.
        if let Err(e) = init_from_yaml(&yaml) {
            panic!("init_from_yaml({}) failed: {}", config_path, e);
        }
    }
    CONFIG
        .get()
        .expect("CONFIG not initialized after init_from_yaml; this should be unreachable")
}

/// Python: setup(config_path)
///
/// - Loads SRS (from cfg.ptau), ONNX (from cfg.onnx)
/// - Runs setup (writes setups + models to cfg.prover paths)
#[pyfunction]
pub fn setup(config_path: &str) -> PyResult<()> {
    let cfg = ensure_cfg(config_path);

    // SRS + ONNX
    let srs: &SRS = &ptau::load_file(
        &cfg.ptau.ptau_path,
        cfg.ptau.pow_len_log,
        cfg.ptau.loaded_pow_len_log,
    );
    let (graph, models_fr) = onnx::load_file(&cfg.onnx.model_path);

    // Prepare model refs for setup
    let model_refs_fr: Vec<&ArrayD<Fr>> = models_fr.iter().map(|x| &x.0).collect();

    // Run setup (with timing)
    let mut timing = TimingTree::default();

    #[cfg(not(feature = "mock_prove"))]
    {
        prover::setup(srs, &graph, &model_refs_fr, &mut timing);
    }

    #[cfg(feature = "mock_prove")]
    {
        let _ = prover::setup(srs, &graph, &model_refs_fr, &mut timing);
    }

    Ok(())
}

/// Python: prove(config_path)
///
/// - Loads SRS + ONNX from cfg
/// - Loads inputs (from JSON if exists, else generates fake inputs)
/// - Runs witness generation
/// - Loads setups + models from disk (paths in config.yaml)
/// - Runs prove (writes encodings + proofs to configured paths)
#[pyfunction]
pub fn prove(config_path: &str) -> PyResult<()> {
    let cfg = ensure_cfg(config_path);

    // SRS + ONNX
    let srs: &SRS = &ptau::load_file(
        &cfg.ptau.ptau_path,
        cfg.ptau.pow_len_log,
        cfg.ptau.loaded_pow_len_log,
    );
    let (mut graph, models_fr) = onnx::load_file(&cfg.onnx.model_path);

    let mut timing = TimingTree::default();

    // Inputs: from JSON if provided, otherwise generated
    let inputs_fr: Vec<ArrayD<Fr>> = if Path::new(&cfg.onnx.input_path).exists() {
        util::load_inputs_from_json_for_onnx(&cfg.onnx.model_path, &cfg.onnx.input_path)
    } else {
        util::generate_fake_inputs_for_onnx(&cfg.onnx.model_path)
    };
    let input_refs_fr: Vec<&ArrayD<Fr>> = inputs_fr.iter().collect();
    let model_refs_fr: Vec<&ArrayD<Fr>> = models_fr.iter().map(|x| &x.0).collect();

    // Witness generation
    let outputs_fr = prover::witness_gen(&input_refs_fr, &graph, &model_refs_fr, &mut timing)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CQ error: {:?}", e)))?;

    // Non-mock path: load setups (projective -> affine) and models from disk
    #[cfg(not(feature = "mock_prove"))]
    {
        // Load setups
        let setups_proj: Vec<(Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>)> =
            CanonicalDeserialize::deserialize_uncompressed_unchecked(
                File::open(&cfg.prover.setup_path).expect("open setup_path"),
            )
            .expect("deserialize setups");

        let setups_affine: Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<DensePolynomial<Fr>>)> =
            util::vec_iter(&setups_proj)
                .map(|x| {
                    (
                        util::vec_iter(&x.0).map(|y| (*y).into()).collect::<Vec<_>>(),
                        util::vec_iter(&x.1).map(|y| (*y).into()).collect::<Vec<_>>(),
                        util::vec_iter(&x.2).map(|y| y.clone()).collect::<Vec<_>>(),
                    )
                })
                .collect();

        let setups_ref: Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>)> =
            setups_affine.iter().map(|x| (&x.0, &x.1, &x.2)).collect();

        // Load encoded models (Data) produced by setup
        let mut models_bytes = Vec::new();
        File::open(&cfg.prover.model_path)
            .expect("open model_path")
            .read_to_end(&mut models_bytes)
            .expect("read model_path");
        let models_data: Vec<ArrayD<Data>> =
            bincode::deserialize::<Vec<ArrayD<Data>>>(&models_bytes).expect("deserialize models");
        let models_data_ref: Vec<&ArrayD<Data>> = models_data.iter().collect();

        // Prove (writes encodings + proofs per config)
        prover::prove(
            srs,
            &input_refs_fr,
            outputs_fr,
            setups_ref,
            models_data_ref,
            &mut graph,
            &mut timing,
        );
    }

    // Mock path: run setup in-memory and use its outputs
    #[cfg(feature = "mock_prove")]
    {
        let (setups_proj, models_data) = prover::setup(srs, &graph, &model_refs_fr, &mut timing);

        let setups_affine: Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<DensePolynomial<Fr>>)> =
            setups_proj
                .iter()
                .map(|(g1p, g2p, polys)| {
                    (
                        g1p.iter().map(|x| (*x).into()).collect::<Vec<_>>(),
                        g2p.iter().map(|x| (*x).into()).collect::<Vec<_>>(),
                        polys.clone(),
                    )
                })
                .collect();

        let setups_ref: Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>)> =
            setups_affine.iter().map(|x| (&x.0, &x.1, &x.2)).collect();

        let models_ref: Vec<&ArrayD<Data>> = models_data.iter().collect();

        prover::prove(
            srs,
            &input_refs_fr,
            outputs_fr,
            setups_ref,
            models_ref,
            &mut graph,
            &mut timing,
        );
    }

    Ok(())
}

/// Python: verify(config_path)
///
/// - Loads SRS + ONNX from cfg
/// - Runs the verifier (reads modelsEnc/inputsEnc/outputsEnc/proofs from disk)
#[pyfunction]
pub fn verify(config_path: &str) -> PyResult<()> {
    let cfg = ensure_cfg(config_path);

    let srs: &SRS = &ptau::load_file(
        &cfg.ptau.ptau_path,
        cfg.ptau.pow_len_log,
        cfg.ptau.loaded_pow_len_log,
    );
    let (graph, _models) = onnx::load_file(&cfg.onnx.model_path);

    let mut timing = TimingTree::default();
    verifier::verify(srs, &graph, &mut timing);
    Ok(())
}

// ---- Python module init ----
#[pymodule]
pub fn pyzktorch(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(setup, m)?)?;
    m.add_function(wrap_pyfunction!(prove, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    Ok(())
}
