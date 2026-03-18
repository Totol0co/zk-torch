// src/python.rs
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyByteArray, PyList, PyMemoryView};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{init_from_yaml};
use crate::onnx;
use crate::ptau;
use crate::graph::Graph;
use crate::basic_block::{SRS, Data, DataEnc};
use crate::util;

use ark_bn254::{Fr, G1Affine, G2Affine};
use ndarray::ArrayD;
use rand::{SeedableRng, rngs::StdRng};
use plonky2::util::timing::TimingTree;
use sha3::{Digest, Keccak256};
use serde::{Serialize, Deserialize};

///  serializer to bytes for ark-serialize types
fn to_vec_bytes<T: CanonicalSerialize>(value: &T) -> PyResult<Vec<u8>> {
    let mut v = Vec::new();
    value.serialize_uncompressed(&mut v)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("serialize: {e:?}")))?;
    Ok(v)
}
fn from_bytes<T: CanonicalDeserialize>(bytes: &[u8]) -> PyResult<T> {
    T::deserialize_uncompressed(bytes)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("deserialize: {e:?}")))
}

/// randomness is seeded from the model, input and output encryption
fn fs_seed_from_encodings<T: ?Sized, U: ?Sized, V: ?Sized>(
    models_enc: &T,
    inputs_enc: &U,
    outputs_enc: &V,
) -> Result<[u8; 32], pyo3::PyErr>
where
    T: serde::Serialize,
    U: serde::Serialize,
    V: serde::Serialize,
{
    let m = bincode::serialize(models_enc)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("bincode modelsEnc: {e}")))?;
    let i = bincode::serialize(inputs_enc)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("bincode inputsEnc: {e}")))?;
    let o = bincode::serialize(outputs_enc)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("bincode outputsEnc: {e}")))?;

    let mut h = Keccak256::new();
    h.update(&m);
    h.update(&i);
    h.update(&o);
    let out = h.finalize();

    let mut seed = [0u8; 32];
    seed.copy_from_slice(&out[..32]);
    Ok(seed)
}


/// Serializable wrapper for `ArrayD<Fr>` using bincode and the arkworks serde helpers
/// (`ark_se` / `ark_de`) to encode/decode field elements.
#[derive(Serialize, Deserialize)]
struct PlainNdArray {
    shape: Vec<usize>,
    #[serde(serialize_with = "crate::util::ark_se", deserialize_with = "crate::util::ark_de")]
    data: Vec<Fr>,
}

// Convert ArrayD<Fr> -> PlainNdArray
fn to_plain_ndarray(a: &ndarray::ArrayD<Fr>) -> PlainNdArray {
    let shape = a.shape().to_vec();
    let data: Vec<Fr> = if let Some(s) = a.as_slice() {
        s.to_vec()
    } else {
        a.iter().copied().collect()
    };
    PlainNdArray { shape, data }
}

// Convert PlainNdArray -> ArrayD<Fr>
fn from_plain_ndarray(p: &PlainNdArray) -> ndarray::ArrayD<Fr> {
    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&p.shape), p.data.clone())
        .expect("PlainNdArray: shape/data mismatch")
}

#[pyclass]
pub struct ZkTorchSession {
    srs: SRS,
    graph: Graph,
    // The `models` tensor list produced by onnx::load_file (used in setup/prove)
    models: Vec<(ArrayD<Fr>, tract_onnx::prelude::DatumType)>,
    // Cache of encoded (Data/DataEnc) models to avoid recompute
    models_data: Vec<ArrayD<Data>>,
    models_enc: Vec<ArrayD<DataEnc>>,
    fold: bool,
    last_inputs_enc: Option<Vec<ArrayD<DataEnc>>>,
    last_outputs_enc: Option<Vec<Vec<ArrayD<DataEnc>>>>,
    setups_aff: Option<Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<ark_poly::univariate::DensePolynomial<Fr>>)>>,
}

#[pymethods]
impl ZkTorchSession {
    /// Initialize from a YAML string (same schema as the config.yaml file).
    /// `fold` must match how you compile the extension 
    #[new]
    pub fn new(config_yaml: &str, fold: bool) -> PyResult<Self> {
        // initialize global CONFIG + LAYER_SETUP_DIR exactly once
        init_from_yaml(config_yaml).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // parse paths & numeric parameters out of CONFIG
        let cfg = crate::CONFIG.get().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("CONFIG not set"))?;
        // 1) Load SRS (KZG powers of tau)
        let srs = ptau::load_file(&cfg.ptau.ptau_path, cfg.ptau.pow_len_log, cfg.ptau.loaded_pow_len_log);  
        // 2) Compile ONNX to ZKTorch DAG + per-block models
        let (graph, models) = onnx::load_file(&cfg.onnx.model_path);                                      

        // Convert models to encoded commitments now (one-time)
        let models_data: Vec<ArrayD<Data>> = models.iter().map(|(m, _)| util::convert_to_data(&srs, m)).collect(); 
        let models_enc: Vec<ArrayD<DataEnc>> = models_data.iter().map(|d| d.map(|x| DataEnc::new(&srs, x))).collect(); 

        Ok(Self {
            srs, graph, models, models_data, models_enc, fold,
            last_inputs_enc: None, last_outputs_enc: None, setups_aff: None,
        })

    }

    
    /// Run setup once, convert to affine, cache it, and return bytes to persist/ship.
    /// NOTE: Make this &mut self to fill the cache.
    pub fn setup(&mut self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let setups_proj = self.graph.setup(&self.srs, &self.models_data.iter().collect());
        let setups_aff: Vec<(Vec<G1Affine>, Vec<G2Affine>, _)> = setups_proj.iter().map(|(g1p,g2p,polys)| {
            (
                g1p.iter().map(|x| (*x).into()).collect(),
                g2p.iter().map(|x| (*x).into()).collect(),
                polys.clone()
            )
        }).collect();
        // cache for reuse in prove()
        self.setups_aff = Some(setups_aff.clone());
        // return bytes (affine) to persist or ship to another machine
        let bytes = to_vec_bytes(&setups_aff)?;
        Ok(PyBytes::new(py, &bytes).into())
    }

    /// Load precomputed setups (affine) from bytes produced by setup()
    /// This enables proving on a different machine without recomputing setup.
    pub fn load_setups_from_bytes(&mut self, py: Python<'_>, any: &PyAny) -> PyResult<()> {
        // generic "read bytes" helper like in verify()/output_verify()
        fn read_bytes<'py>(_py: Python<'py>, any: &PyAny) -> PyResult<Vec<u8>> {
            if let Ok(b) = any.downcast::<PyBytes>() { return Ok(b.as_bytes().to_vec()); }
            if let Ok(v) = any.extract::<Vec<u8>>() { return Ok(v); }
            if let Ok(lst) = any.downcast::<pyo3::types::PyList>() {
                let mut v = Vec::with_capacity(lst.len());
                for it in lst.iter() { v.push(it.extract::<u8>()?); }
                return Ok(v);
            }
            Err(pyo3::exceptions::PyTypeError::new_err(
                "expected bytes, bytearray, memoryview, list[int], or any buffer-compatible object",
            ))
        }
        let bytes = read_bytes(py, any)?;
        // Deserialize affine setups: Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<DensePolynomial<Fr>>)>
        let setups_aff: Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<ark_poly::univariate::DensePolynomial<Fr>>)>
            = from_bytes(&bytes)?;
        self.setups_aff = Some(setups_aff);
        Ok(())
    }



    /// Prove with  JSON input file. If None, random inputs are generated.
    /// Returns (proofs_bytes, acc_proofs_bytes, output_bytes)
    pub fn prove(&mut self, py: Python<'_>, input_json: Option<&str>) -> PyResult<(Py<PyBytes>, Option<Py<PyBytes>>, Py<PyBytes>)> {

        // 1) Build inputs for the ONNX graph (either from JSON or generate)
        let cfg = crate::CONFIG.get().unwrap();
        let inputs_fr: Vec<ArrayD<Fr>> = if let Some(p) = input_json {
            util::load_inputs_from_json_for_onnx(&cfg.onnx.model_path, p)
        } else {
            util::generate_fake_inputs_for_onnx(&cfg.onnx.model_path)
        };

        // 2) Forward pass in finite field to obtain outputs (witness tensors)
        let input_refs: Vec<&ArrayD<Fr>> = inputs_fr.iter().collect();
        let model_refs: Vec<&ArrayD<Fr>> = self.models.iter().map(|(m, _)| m).collect();
        let outputs_fr = self.graph.run(&input_refs, &model_refs)
             .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CQ range error at input={:?}", e.input)))?; 
        
        // Sorties finales CLaires (sinks) du modèle
        let model_outputs_fr: Vec<ArrayD<Fr>> = self.graph.collect_model_outputs(&outputs_fr);
        eprintln!("[prove] final outputs (self.outputs) = {}", self.graph.outputs.len());
        for (k, a) in model_outputs_fr.iter().enumerate() {
            eprintln!("  final[{}].shape = {:?}", k, a.shape());
        }
        let sinks_plain: Vec<PlainNdArray> = model_outputs_fr.iter().map(|a| to_plain_ndarray(a)).collect();
        let model_outputs_bytes = bincode::serialize(&sinks_plain)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("bincode model_outputs_fr: {e}")))?;


        // 3) Encode inputs & outputs to KZG-committed vectors (Data)
        let inputs_data: Vec<ArrayD<Data>> = inputs_fr.iter().map(|m| util::convert_to_data_public(&self.srs, m)).collect();
        let inputs_data_enc: Vec<ArrayD<DataEnc>> =
            inputs_data.iter().map(|d| d.map(|x| DataEnc::new(&self.srs, x))).collect();
        let input_data_refs: Vec<&ArrayD<Data>> = inputs_data.iter().collect();
        
         let outputs_refs_owned_fr: Vec<Vec<&ArrayD<Fr>>> =
             outputs_fr.iter().map(|v| v.iter().collect::<Vec<_>>()).collect();
         let outputs_refs_fr: Vec<&Vec<&ArrayD<Fr>>> =
             outputs_refs_owned_fr.iter().collect();

        let mut timing = TimingTree::new("encodeOutputs", log::Level::Info);
        let outputs_data = self.graph.encodeOutputs(
            &self.srs,
            &self.models_data.iter().collect(),
            &input_data_refs,
            &outputs_refs_fr,
            &mut timing
        );

        
        let outputs_data_public: Vec<Vec<ArrayD<Data>>> =
            outputs_data.iter()
                .map(|vv| vv.iter()
                    .map(|darr| darr.map(|d| Data::new_public(&self.srs, &d.raw)))
                    .collect()
                ).collect();


        let outputs_enc: Vec<Vec<ArrayD<DataEnc>>> =
            outputs_data_public.iter().map(|vv| vv.iter().map(|d| d.map(|x| DataEnc::new(&self.srs, x))).collect()).collect();
        self.last_inputs_enc = Some(inputs_data_enc.clone());
        self.last_outputs_enc = Some(outputs_enc.clone());

        
        // 4) reuse setups computed or loaded earlier
        let setups_aff = self.setups_aff.as_ref().ok_or_else(||
            pyo3::exceptions::PyRuntimeError::new_err("No setups cached. Call setup() or load_setups_from_bytes() before prove().")
        )?;
        let setups_refs: Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<ark_poly::univariate::DensePolynomial<Fr>>)>
            = setups_aff.iter().map(|(a,b,c)| (a,b,c)).collect();


        // 5) Prove
        let seed = fs_seed_from_encodings(&self.models_enc, &inputs_data_enc, &outputs_enc)?;
        let mut rng = StdRng::from_seed(seed);

        let mut prove_timing = TimingTree::new("prove", log::Level::Info);

        
        #[cfg(not(feature = "fold"))]
        {
            let outputs_refs_owned: Vec<Vec<&ArrayD<Data>>> =
                outputs_data_public.iter().map(|vv| vv.iter().collect::<Vec<_>>()).collect(); // Build Vec<Vec<&ArrayD<Data>>> then &Vec<&Vec<&ArrayD<Data>>>
            let outputs_refs: Vec<&Vec<&ArrayD<Data>>> =
                outputs_refs_owned.iter().collect();


            let proofs = self.graph.prove(
                &self.srs,
                &setups_refs,
                &self.models_data.iter().collect(),
                &input_data_refs,
                &outputs_refs,
                &mut rng,
                &mut prove_timing
            );
            
            let pbytes = to_vec_bytes(&proofs)?;
            let out_plain = PyBytes::new(py, &model_outputs_bytes).into();
            return Ok((PyBytes::new(py, &pbytes).into(), None, out_plain));

         }


        
        #[cfg(feature = "fold")]
        {
            let outputs_refs_owned: Vec<Vec<&ArrayD<Data>>> =
                outputs_data_public.iter().map(|vv| vv.iter().collect::<Vec<_>>()).collect();
            let outputs_refs: Vec<&Vec<&ArrayD<Data>>> =
                outputs_refs_owned.iter().collect();

            let (proofs, acc_proofs) = self.graph.prove(
                &self.srs,
                &setups_refs,
                &self.models_data.iter().collect(),
                &input_data_refs,
                &outputs_refs,
                &mut rng,
                &mut prove_timing
            );
             
            let pbytes = to_vec_bytes(&proofs)?;
            let abytes = to_vec_bytes(&acc_proofs)?;
            let out_plain = PyBytes::new(py, &model_outputs_bytes).into();
            return Ok((PyBytes::new(py, &pbytes).into(), Some(PyBytes::new(py, &abytes).into()), out_plain));

        }

    }

    /// Verify proofs (and accumulator proofs if built with `fold`)
    pub fn verify(&self, py: Python<'_>, proofs: &PyAny, acc_proofs: Option<&PyAny>) -> PyResult<bool> {

        let mut timing = TimingTree::new("verify", log::Level::Info);
        
        let inputs_enc = self.last_inputs_enc.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("verify() requires prior prove()"))?;
        let outputs_enc = self.last_outputs_enc.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("verify() requires prior prove()"))?;

        // Derive the same FS seed as in `prove()`
        let seed = fs_seed_from_encodings(&self.models_enc, inputs_enc, outputs_enc)?;
        let mut rng = StdRng::from_seed(seed);
        
        // helper to read a Python object into Vec<u8> 
        fn read_bytes<'py>(_py: Python<'py>, any: &PyAny) -> PyResult<Vec<u8>> {
            // 1) bytes
            if let Ok(b) = any.downcast::<PyBytes>() {                
                return Ok(b.as_bytes().to_vec());
            }
            // 2) generic buffer objects (bytearray, memoryview, numpy arrays, etc.)
            if let Ok(v) = any.extract::<Vec<u8>>() {
                return Ok(v);
            }
            // 3) list[int] fallback
            if let Ok(lst) = any.downcast::<PyList>() {
                let mut v = Vec::with_capacity(lst.len());
                for item in lst.iter() {
                    v.push(item.extract::<u8>()?);
                }
                return Ok(v);
            }
            Err(pyo3::exceptions::PyTypeError::new_err(
                "expected bytes, bytearray, memoryview, list[int], or any buffer-compatible object",
            ))
        }

        #[cfg(not(feature = "fold"))]
        {
            // Deserialize proofs
            let proofs_vec: Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)> =
                from_bytes(&read_bytes(py, proofs)?)?;
            let proofs_refs: Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)> =
                proofs_vec.iter().map(|(a,b,c)| (a,b,c)).collect();

            // Build model / input / output encoded refs
            let model_enc_refs: Vec<&ArrayD<DataEnc>> = self.models_enc.iter().collect();
            let inputs_enc = self.last_inputs_enc.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("verify() requires a prior prove() in this session (no cached encoded inputs)"))?;
            let outputs_enc = self.last_outputs_enc.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("verify() requires a prior prove() in this session (no cached encoded outputs)"))?;
            let inputs_enc_refs: Vec<&ArrayD<DataEnc>> = inputs_enc.iter().collect();
            let outputs_refs_owned_enc: Vec<Vec<&ArrayD<DataEnc>>> =
                outputs_enc.iter().map(|vv| vv.iter().collect::<Vec<_>>()).collect();
            let outputs_enc_refs: Vec<&Vec<&ArrayD<DataEnc>>> =
                outputs_refs_owned_enc.iter().collect();

            // Verify
            self.graph.verify(
                &self.srs,
                &model_enc_refs,
                &inputs_enc_refs,
                &outputs_enc_refs,
                &proofs_refs,
                &mut rng,
                &mut timing
            );
            return Ok(true); // if no panic/assert, it verified
        }


        #[cfg(feature = "fold")]
        {
            // Deserialize proofs / acc-proofs
            let proofs_vec: Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)> =
                from_bytes(&read_bytes(py, proofs)?)?;
            let acc_py = acc_proofs.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("acc_proofs required with fold"))?;
            let acc_vec: Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<_>)> =
                from_bytes(&read_bytes(py, acc_py)?)?;
            let proofs_refs: Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>)> =
                proofs_vec.iter().map(|(a,b,c)| (a,b,c)).collect();
            let acc_refs: Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>, &Vec<_>)> =
                acc_vec.iter().map(|(a,b,c,d)| (a,b,c,d)).collect();

            // Build model / input / output encoded refs
            let model_enc_refs: Vec<&ArrayD<DataEnc>> = self.models_enc.iter().collect();
            let inputs_enc = self.last_inputs_enc.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("verify() requires a prior prove() in this session (no cached encoded inputs)"))?;
            let outputs_enc = self.last_outputs_enc.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("verify() requires a prior prove() in this session (no cached encoded outputs)"))?;
            let inputs_enc_refs: Vec<&ArrayD<DataEnc>> = inputs_enc.iter().collect();
            let outputs_refs_owned_enc: Vec<Vec<&ArrayD<DataEnc>>> =
                outputs_enc.iter().map(|vv| vv.iter().collect::<Vec<_>>()).collect();
            let outputs_enc_refs: Vec<&Vec<&ArrayD<DataEnc>>> =
                outputs_refs_owned_enc.iter().collect();

            // Verify
            let (_final_proofs_idx, _final_acc_idx) = self.graph.verify(
                &self.srs,
                &model_enc_refs,
                &inputs_enc_refs,
                &outputs_enc_refs,
                &proofs_refs,
                &acc_refs,
                &mut rng,
                &mut timing
            ); 
            Ok(true)

        }
    }

    /// Return True iff `input_json` encodes to the exact same commitments inputsEnc as the inputs used by the most recent `prove()` call
    pub fn input_verify(&self, input_json: &str) -> PyResult<bool> {
        // 0) Must have proved at least once in this session
        let cached = self.last_inputs_enc.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "No cached inputsEnc; call prove() first in this session"
            )
        })?;

        // 1) Rebuild the inputs from JSON with the same ONNX pipeline
        let cfg = crate::CONFIG.get().unwrap();
        let inputs_fr: Vec<ArrayD<Fr>> =
            crate::util::load_inputs_from_json_for_onnx(&cfg.onnx.model_path, input_json);

        // 2) Encode to Data -> DataEnc using the PUBLIC encoder (r = 0 !)
        let inputs_data: Vec<ArrayD<Data>> =
            inputs_fr.iter().map(|m| crate::util::convert_to_data_public(&self.srs, m)).collect();

        let inputs_enc_new: Vec<ArrayD<DataEnc>> =
            inputs_data.iter().map(|d| d.map(|x| DataEnc::new(&self.srs, x))).collect();

        // 3) Pairwise commitment equality check (byte-for-byte)
        if inputs_enc_new.len() != cached.len() {
            return Ok(false);
        }
        for (a, b) in inputs_enc_new.iter().zip(cached.iter()) {
            let ab = bincode::serialize(a).map_err(|e|
                pyo3::exceptions::PyRuntimeError::new_err(format!("bincode new inputsEnc: {e}"))
            )?;
            let bb = bincode::serialize(b).map_err(|e|
                pyo3::exceptions::PyRuntimeError::new_err(format!("bincode cached inputsEnc: {e}"))
            )?;
            if ab != bb {
                return Ok(false);
            }
        }
        Ok(true)
    }
        
    /// Verify that provided plaintext final outputs match the cached public commitments
    /// produced by the most recent `prove()` call in this session.
    pub fn output_verify(&self, py: Python<'_>, out_plain_bytes: &PyAny) -> PyResult<bool> {
        // Require cached encoded outputs from the last prove()
        let outputs_cached = self.last_outputs_enc.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("No cached outputsEnc; call prove() first in this session")
        })?;

        // Helper: read a Python object as raw bytes (bytes/bytearray/memoryview/list[int])
        fn read_bytes<'py>(_py: Python<'py>, any: &PyAny) -> PyResult<Vec<u8>> {
            if let Ok(b) = any.downcast::<PyBytes>() { return Ok(b.as_bytes().to_vec()); }
            if let Ok(v) = any.extract::<Vec<u8>>() { return Ok(v); }
            if let Ok(lst) = any.downcast::<pyo3::types::PyList>() {
                let mut v = Vec::with_capacity(lst.len());
                for item in lst.iter() { v.push(item.extract::<u8>()?); }
                return Ok(v);
            }
            Err(pyo3::exceptions::PyTypeError::new_err(
                "expected bytes, bytearray, memoryview, list[int], or any buffer-compatible object",
            ))
        }

        // Deserialize plaintext final outputs (bincode Vec<PlainNdArray>) -> Vec<ArrayD<Fr>>
        let out_bytes = read_bytes(py, out_plain_bytes)?;
        let sinks_plain_wrapped: Vec<PlainNdArray> = bincode::deserialize(&out_bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bincode outputs_plain: {e}")))?;
        let sinks_plain: Vec<ArrayD<Fr>> = sinks_plain_wrapped.iter().map(|p| from_plain_ndarray(p)).collect();

        // Build a usage map (kept for topology reference; not used below since we rely on graph.outputs)
        let n = self.graph.nodes.len();
        let mut used: Vec<Vec<bool>> = (0..n)
            .map(|i| vec![false; outputs_cached[i].len()])
            .collect();
        for node in &self.graph.nodes {
            for (bb_idx, out_idx) in node.inputs.iter() {
                if *bb_idx >= 0 {
                    let i = *bb_idx as usize;
                    let j = *out_idx;
                    if i < used.len() && j < used[i].len() {
                        used[i][j] = true;
                    }
                }
            }
        }

        // Collect cached final (sink) commitments from the last prove(), using canonical graph outputs.
        // Skip outputs of nodes that were precomputable (not encoded/proved).
        let mut cached_sink_encs: Vec<&ArrayD<DataEnc>> = Vec::new();
        for (node_idx, out_idx) in &self.graph.outputs {
            if *node_idx >= 0 {
                let i = *node_idx as usize;
                let j = *out_idx;
                if i < outputs_cached.len() && j < outputs_cached[i].len() && !self.graph.precomputable.encodeOutputs[i] {
                    cached_sink_encs.push(&outputs_cached[i][j]);
                }
            }
        }

        // The number of provided plaintext sinks must match the number of cached sink commitments
        if sinks_plain.len() != cached_sink_encs.len() {
            eprintln!("[output_verify] mismatch count: plain={} cached_sinks={}", sinks_plain.len(), cached_sink_encs.len());
            return Ok(false);
        }

        // Re-encode each plaintext sink as PUBLIC (r=0), then compare bincode blobs byte-for-byte
        for (k, (plain_arr, cached_enc)) in sinks_plain.iter().zip(cached_sink_encs.into_iter()).enumerate() {
            // Plain Fr -> Data (public) -> DataEnc
            let data_pub = crate::util::convert_to_data_public(&self.srs, plain_arr);
            let enc_new = data_pub.map(|x| DataEnc::new(&self.srs, x));

            // Serialize both new and cached encodings and compare exact bytes
            let ab = bincode::serialize(&enc_new)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("bincode new sink {}: {e}", k)))?;
            let bb = bincode::serialize(cached_enc)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("bincode cached sink {}: {e}", k)))?;

            if ab != bb {
                eprintln!("[output_verify] sink {} mismatch (hash new != cached)", k);
                return Ok(false);
            }
        }

        // All sinks matched the cached commitments
        Ok(true)
    }

    /// Returns a Python list of tuples: (shape: List[int], flat_data_ints: List[int])
    /// It decodes the bincode 'out_plain' (Vec<PlainNdArray>) and maps Fr -> signed int via fr_to_int.
    pub fn decode_outputs_plain_to_raw(&self, py: pyo3::Python<'_>, out_plain_bytes: &pyo3::types::PyAny)
        -> pyo3::PyResult<Vec<(Vec<usize>, Vec<i64>)>>
    {
        // Read bytes from Python object (bytes/bytearray/memoryview/list[int])
        fn read_bytes<'py>(_py: pyo3::Python<'py>, any: &pyo3::types::PyAny) -> pyo3::PyResult<Vec<u8>> {
            if let Ok(b) = any.downcast::<pyo3::types::PyBytes>() { return Ok(b.as_bytes().to_vec()); }
            if let Ok(v) = any.extract::<Vec<u8>>() { return Ok(v); }
            if let Ok(lst) = any.downcast::<pyo3::types::PyList>() {
                let mut v = Vec::with_capacity(lst.len());
                for item in lst.iter() { v.push(item.extract::<u8>()?); }
                return Ok(v);
            }
            Err(pyo3::exceptions::PyTypeError::new_err(
                "expected bytes, bytearray, memoryview, list[int], or any buffer-compatible object",
            ))
        }

        let bytes = read_bytes(py, out_plain_bytes)?;
        let wrapped: Vec<PlainNdArray> = bincode::deserialize(&bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bincode: {e}")))?;

        // Convert to (shape, flat_ints) using util::fr_to_int
        let mut out: Vec<(Vec<usize>, Vec<i64>)> = Vec::with_capacity(wrapped.len());
        for p in wrapped.iter() {
            let arr_fr = from_plain_ndarray(p);
            let ints: Vec<i64> = arr_fr.iter().map(|fr| crate::util::fr_to_int(*fr) as i64).collect();
            out.push((p.shape.clone(), ints));
        }
        Ok(out)
    }

    /// Expose the scale_factor_log from CONFIG (used to dequantize float outputs)
    pub fn get_scale_factor_log(&self) -> pyo3::PyResult<usize> {
        let cfg = crate::CONFIG.get().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("CONFIG not set"))?;
        Ok(cfg.sf.scale_factor_log)
    }
    
    
    pub fn encode_outputs_plain_from_raw(
        &self,
        py: Python<'_>,                    
        shapes: Vec<Vec<usize>>,
        flat_ints: Vec<Vec<i64>>,
    ) -> PyResult<Py<PyBytes>> {
        if shapes.len() != flat_ints.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("shapes.len()={} != flat_ints.len()={}", shapes.len(), flat_ints.len())
            ));
        }

        // Build the PlainNdArray list
        let mut vec_plain = Vec::<PlainNdArray>::with_capacity(shapes.len());
        for (shape, ints) in shapes.into_iter().zip(flat_ints.into_iter()) {
            let data_fr: Vec<Fr> = ints.into_iter().map(Fr::from).collect();
            vec_plain.push(PlainNdArray { shape, data: data_fr });
        }

        // Serialize to bincode
        let bytes = bincode::serialize(&vec_plain)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("bincode serialize: {e}")))?;

        Ok(PyBytes::new(py, &bytes).into())
    }

}

#[pymodule]
fn pyzktorch(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<ZkTorchSession>()?;
    Ok(())
}