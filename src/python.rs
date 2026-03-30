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
        if let Err(e) = init_from_yaml(&yaml) {
            panic!("init_from_yaml({}) failed: {}", config_path, e);
        }
    }
    CONFIG
        .get()
        .expect("CONFIG not initialized after init_from_yaml")
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

    // Run setup 
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

    // load setups (projective -> affine) and models from disk
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

use serde_json::Value;

use crate::basic_block::{DataEnc};

use ndarray::{IxDyn};

/// Internal: load a file into Vec<u8>
fn read_all_bytes(p: &str) -> Result<Vec<u8>, String> {
    let mut b = Vec::new();
    File::open(p).map_err(|e| format!("open {}: {}", p, e))?
        .read_to_end(&mut b).map_err(|e| format!("read {}: {}", p, e))?;
    Ok(b)
}

/// Internal: try verifier paths first, then prover paths
fn first_existing_path(primary: &str, fallback: &str) -> String {
    if Path::new(primary).exists() { primary.to_string() } else { fallback.to_string() }
}

/// Internal: encode public ArrayD<Fr> -> ArrayD<DataEnc>
fn to_public_dataenc(srs: &SRS, arr: &ArrayD<Fr>) -> ArrayD<DataEnc> {
    let d = util::convert_to_data_public(srs, arr);               // r = 0 (public)
    d.map(|x| DataEnc::new(srs, x))                               // deterministic DataEnc
}

/// Internal: deep-equals two Vec<ArrayD<DataEnc>>
fn dataenc_vec_equal(a: &Vec<ArrayD<DataEnc>>, b: &Vec<ArrayD<DataEnc>>) -> bool {
    if a.len() != b.len() { return false; }
    for (aa, bb) in a.iter().zip(b.iter()) {
        if aa.shape() != bb.shape() { return false; }
        // DataEnc derives PartialEq
        if aa != bb { return false; }
    }
    true
}

/// verify that a given JSON input file matches the public inputsEnc on disk.
#[pyfunction]
pub fn verify_public_inputs(config_path: &str, json_input_path: &str) -> PyResult<bool> {
    // Ensure CONFIG (and dirs) are initialized once
    let cfg = ensure_cfg(config_path);


    // Load SRS
    let srs: &SRS = &ptau::load_file(
        &cfg.ptau.ptau_path,
        cfg.ptau.pow_len_log,
        cfg.ptau.loaded_pow_len_log,
    );

    // Use the same loader used by the prover 
    let inputs_fr: Vec<ArrayD<Fr>> =
        util::load_inputs_from_json_for_onnx(&cfg.onnx.model_path, json_input_path);

    // Encode as PUBLIC encodings (r=0), then DataEnc
    let local_inputs_enc: Vec<ArrayD<DataEnc>> =
        inputs_fr.iter().map(|a| to_public_dataenc(srs, a)).collect();

    // Load inputsEnc from disk 
    let enc_path = first_existing_path(&cfg.verifier.enc_input_path, &cfg.prover.enc_input_path);
    let bytes = read_all_bytes(&enc_path).map_err(|e|
        pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let disk_inputs_enc: Vec<ArrayD<DataEnc>> =
        bincode::deserialize(&bytes).map_err(|e|
            pyo3::exceptions::PyRuntimeError::new_err(format!("deserialize {}: {}", enc_path, e)))?;

    Ok(dataenc_vec_equal(&local_inputs_enc, &disk_inputs_enc))
}

/// Parse a JSON array-of-arrays (or single array) into ArrayD<Fr> of FIELD INTS (no scaling).
fn parse_field_tensors_from_json(json_path: &str) -> Result<Vec<ArrayD<Fr>>, String> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(json_path)
        .map_err(|e| format!("read {}: {}", json_path, e))?)
        .map_err(|e| format!("parse {}: {}", json_path, e))?;
    fn to_fr_array(v: &Value) -> Result<(Vec<Fr>, Vec<usize>), String> {
        fn shape_of(v: &Value) -> Result<Vec<usize>, String> {
            match v {
                Value::Array(a) => {
                    if a.is_empty() { return Ok(vec![0]); }
                    let mut sub = shape_of(&a[0])?;
                    for elem in a.iter().skip(1) {
                        if shape_of(elem)? != sub { return Err("ragged array".into()); }
                    }
                    let mut out = vec![a.len()];
                    out.extend(sub);
                    Ok(out)
                }
                Value::Number(_) => Ok(vec![]),
                _ => Err("expected array/number".into())
            }
        }
        fn flatten(v: &Value, out: &mut Vec<Fr>) -> Result<(), String> {
            match v {
                Value::Array(a) => { for e in a { flatten(e, out)?; } Ok(()) }
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() { out.push(Fr::from(i)); Ok(()) }
                    else { Err("non-integer number in 'field' mode".into()) }
                }
                _ => Err("expected array/number".into())
            }
        }
        let shp = shape_of(v)?;
        let mut data = Vec::new();
        flatten(v, &mut data)?;
        Ok((data, shp))
    }
    // shape+data object -> ArrayD<Fr>
    fn object_shape_data_to_arr(obj: &Value) -> Result<ArrayD<Fr>, String> {
        let shape_v = obj.get("shape").ok_or("missing 'shape'")?;
        let data_v  = obj.get("data").ok_or("missing 'data'")?;
        let shape: Vec<usize> = shape_v.as_array()
            .ok_or("'shape' must be array")?
            .iter()
            .map(|x| x.as_u64().ok_or("shape elt not u64").map(|u| u as usize))
            .collect::<Result<_,_>>()?;
        let data: Vec<Fr> = data_v.as_array()
            .ok_or("'data' must be array")?
            .iter()
            .map(|x| x.as_i64().ok_or("data elt not i64").map(Fr::from))
            .collect::<Result<_,_>>()?;
        ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| format!("shape/data mismatch: {e}"))
    }

    let tensors: Vec<Value> = match &v {
        Value::Object(m) => {
            if let Some(x) = m.get("output_data") { vec![x.clone()] }
            else if let Some(x) = m.get("outputs") {
                if let Value::Array(arr) = x { arr.clone() } else { vec![x.clone()] }
            } else { return Err("expected keys 'output_data' or 'outputs'".into()); }
        }
        Value::Array(_) => vec![v.clone()],
        _ => return Err("expected object or array".into()),
    };
    
    let mut out = Vec::new();
    for t in tensors {
        match &t {
            Value::Object(obj) if obj.get("shape").is_some() && obj.get("data").is_some() => {
                out.push(object_shape_data_to_arr(&t)?);
            }
            _ => {
                let (flat, shp) = to_fr_array(&t)?;
                let arr = ArrayD::from_shape_vec(IxDyn(&shp), flat)
                    .map_err(|e| format!("ndarray build: {e}"))?;
                out.push(arr);
            }
        }
    }

    Ok(out)
}

/// Like parse_field_tensors_from_json, but treats JSON as floats and applies scale factor
fn parse_float_tensors_from_json_scaled(json_path: &str, sf_log2: usize) -> Result<Vec<ArrayD<Fr>>, String> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(json_path)
        .map_err(|e| format!("read {}: {}", json_path, e))?)
        .map_err(|e| format!("parse {}: {}", json_path, e))?;
    fn to_fr_array_scaled(v: &Value, sf: f32) -> Result<(Vec<Fr>, Vec<usize>), String> {
        fn shape_of(v: &Value) -> Result<Vec<usize>, String> {
            match v {
                Value::Array(a) => {
                    if a.is_empty() { return Ok(vec![0]); }
                    let mut sub = shape_of(&a[0])?;
                    for elem in a.iter().skip(1) {
                        if shape_of(elem)? != sub { return Err("ragged array".into()); }
                    }
                    let mut out = vec![a.len()];
                    out.extend(sub);
                    Ok(out)
                }
                Value::Number(_) => Ok(vec![]),
                _ => Err("expected array/number".into())
            }
        }
        fn flatten(v: &Value, out: &mut Vec<Fr>, sf: f32) -> Result<(), String> {
            match v {
                Value::Array(a) => { for e in a { flatten(e, out, sf)?; } Ok(()) }
                Value::Number(n) => {
                    // Treat as f64 -> apply scale -> round
                    let f = n.as_f64().ok_or("non-numeric value")? as f32;
                    let y = (f * sf).round();
                    // Clamp 
                    out.push(Fr::from(y as i64));
                    Ok(())
                }
                _ => Err("expected array/number".into())
            }
        }
        let shp = shape_of(v)?;
        let mut data = Vec::new();
        flatten(v, &mut data, sf)?;
        Ok((data, shp))
    }
    //  shape+data with FLOATS -> quantize back using scale and build ArrayD<Fr>
    fn object_shape_data_to_arr_scaled(obj: &Value, sf: f32) -> Result<ArrayD<Fr>, String> {
        let shape_v = obj.get("shape").ok_or("missing 'shape'")?;
        let data_v  = obj.get("data").ok_or("missing 'data'")?;
        let shape: Vec<usize> = shape_v.as_array()
            .ok_or("'shape' must be array")?
            .iter()
            .map(|x| x.as_u64().ok_or("shape elt not u64").map(|u| u as usize))
            .collect::<Result<_,_>>()?;
        let data: Vec<Fr> = data_v.as_array()
            .ok_or("'data' must be array")?
            .iter()
            .map(|x| {
                let f = x.as_f64().ok_or("data elt not float")? as f32;
                let q = (f * sf).round();
                Ok(Fr::from(q as i64))
            })
            .collect::<Result<_,String>>()?;
        ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| format!("shape/data mismatch: {e}"))
    }

    let tensors: Vec<Value> = match &v {
        Value::Object(m) => {
            if let Some(x) = m.get("output_data") { vec![x.clone()] }
            else if let Some(x) = m.get("outputs") {
                if let Value::Array(arr) = x { arr.clone() } else { vec![x.clone()] }
            } else { return Err("expected keys 'output_data' or 'outputs'".into()); }
        }
        Value::Array(_) => vec![v.clone()],
        _ => return Err("expected object or array".into()),
    };
    let sf = (1usize << sf_log2) as f32;
    let mut out = Vec::new();
    for t in tensors {
        match &t {
            Value::Object(obj) if obj.get("shape").is_some() && obj.get("data").is_some() => {
                out.push(object_shape_data_to_arr_scaled(&t, sf)?);
            }
            _ => {
                let (flat, shp) = to_fr_array_scaled(&t, sf)?;
                let arr = ArrayD::from_shape_vec(IxDyn(&shp), flat)
                    .map_err(|e| format!("ndarray build: {e}"))?;
                out.push(arr);
            }
        }

    }
    Ok(out)
}

/// Python: verify that a given JSON output matches the public outputsEnc on disk.
/// `mode` = "field" (integers already in the field) or "float" (apply 2^scale_factor_log & round).
#[pyfunction]
pub fn verify_public_outputs(config_path: &str, json_output_path: &str, mode: &str) -> PyResult<bool> {
    // 1) Config + SRS
    let cfg = ensure_cfg(config_path);
    let srs: &SRS = &ptau::load_file(
        &cfg.ptau.ptau_path,
        cfg.ptau.pow_len_log,
        cfg.ptau.loaded_pow_len_log,
    );

    // 2) Parse the JSON of final outputs → Vec<ArrayD<Fr>>
    let outs_fr: Vec<ArrayD<Fr>> = match mode {
        "field" => parse_field_tensors_from_json(json_output_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?,
        "float" => parse_float_tensors_from_json_scaled(json_output_path, cfg.sf.scale_factor_log)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?,
        _ => return Err(pyo3::exceptions::PyValueError::new_err("mode must be 'field' or 'float'")),
    };

    // 3) Encode JSON as PUBLIC commitments (r=0) → Vec<ArrayD<DataEnc>>
    let local_outs_enc: Vec<ArrayD<DataEnc>> =
        outs_fr.iter().map(|a| to_public_dataenc(srs, a)).collect();

    // 4) Rebuild the graph to know which node outputs are the final ones
    let (graph, _models) = onnx::load_file(&cfg.onnx.model_path);

    // 5) Read the written outputsEnc and extract the finals
    let enc_path = if Path::new(&cfg.verifier.enc_output_path).exists() {
        cfg.verifier.enc_output_path.as_str()
    } else {
        cfg.prover.enc_output_path.as_str()
    };
    let bytes = read_all_bytes(enc_path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let all_outs_enc: Vec<Vec<ArrayD<DataEnc>>> = bincode::deserialize(&bytes)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("deserialize {}: {}", enc_path, e)
        ))?;

    // Extract only the finals as recorded in graph.outputs
    let mut disk_finals: Vec<ArrayD<DataEnc>> = Vec::new();
    for (node_idx, port) in &graph.outputs {
        if *node_idx >= 0 {
            let i = *node_idx as usize;
            let j = *port;
            if i >= all_outs_enc.len() || j >= all_outs_enc[i].len() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("outputsEnc index out of range: node {} port {}", i, j)
                ));
            }
            disk_finals.push(all_outs_enc[i][j].clone());
        }
    }

    // 6) Compare deterministic public commitments
    Ok(dataenc_vec_equal(&local_outs_enc, &disk_finals))
}



// ---- Python module init ----
#[pymodule]
pub fn pyzktorch(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(setup, m)?)?;
    m.add_function(wrap_pyfunction!(prove, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;

    m.add_function(wrap_pyfunction!(verify_public_inputs, m)?)?;
    m.add_function(wrap_pyfunction!(verify_public_outputs, m)?)?;

    Ok(())
}
