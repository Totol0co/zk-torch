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

/// A simple serializer to bytes for any ark-serialize type.
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


use sha3::{Digest, Keccak256};

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

}

#[pymethods]
impl ZkTorchSession {
    /// Initialize from a YAML string (same schema as the CLI config).
    /// `fold` must match how you compile the extension (see build section).
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
            last_inputs_enc: None, last_outputs_enc: None,
        })

    }

    /// Run setup for all blocks in the DAG; returns a serialized Vec of (G1,G2,poly) tuples.
    
    /// Run setup for all blocks in the DAG; returns serialized bytes.
    pub fn setup(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let setups = self.graph.setup(&self.srs, &self.models_data.iter().collect());                     
        let bytes = to_vec_bytes(&setups)?;
        Ok(PyBytes::new(py, &bytes).into())
     }


    /// Prove with optional JSON input file. If None, random inputs are generated (same as CLI behaviour).
    /// Returns (proofs_bytes, acc_proofs_bytes_or_none)
    pub fn prove(&mut self, py: Python<'_>, input_json: Option<&str>) -> PyResult<(Py<PyBytes>, Option<Py<PyBytes>>)> {
        // 1) Build inputs for the ONNX graph (either from JSON or generate)
        
        let cfg = crate::CONFIG.get().unwrap();
        // util::onnx module exposes filename-only helpers
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

        // 3) Encode inputs & outputs to KZG-committed vectors (Data)
        
        let inputs_data: Vec<ArrayD<Data>> = inputs_fr.iter().map(|m| util::convert_to_data(&self.srs, m)).collect();
        let inputs_data_enc: Vec<ArrayD<DataEnc>> =
            inputs_data.iter().map(|d| d.map(|x| DataEnc::new(&self.srs, x))).collect();

        let input_data_refs: Vec<&ArrayD<Data>> = inputs_data.iter().collect();
        
        
        // DAG node outputs are Vec<Vec<ArrayD<Fr>>>; encodeOutputs wants &Vec<&Vec<&ArrayD<Fr>>>. Build an owned container first.
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

        let outputs_enc: Vec<Vec<ArrayD<DataEnc>>> =
            outputs_data.iter().map(|vv| vv.iter().map(|d| d.map(|x| DataEnc::new(&self.srs, x))).collect()).collect();
        self.last_inputs_enc = Some(inputs_data_enc.clone());
        self.last_outputs_enc = Some(outputs_enc.clone());

        // 4) Setup (affine) for all BBs
        let setups_proj = self.graph.setup(&self.srs, &self.models_data.iter().collect());
        // Convert projective to affine for the prover/ verifier APIs:
        let setups_aff: Vec<(Vec<G1Affine>, Vec<G2Affine>, _)> = setups_proj.iter().map(|(g1p, g2p, polys)| {
            (g1p.iter().map(|x| (*x).into()).collect(), g2p.iter().map(|x| (*x).into()).collect(), polys.clone())
        }).collect();
        let setups_refs: Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &_)> = setups_aff.iter().map(|(a,b,c)| (a,b,c)).collect();

        // 5) Prove
        
        let seed = fs_seed_from_encodings(&self.models_enc, &inputs_data_enc, &outputs_enc)?;
        let mut rng = StdRng::from_seed(seed);

        let mut prove_timing = TimingTree::new("prove", log::Level::Info);

        
        #[cfg(not(feature = "fold"))]
        {
            // Build Vec<Vec<&ArrayD<Data>>> then &Vec<&Vec<&ArrayD<Data>>>
            
            let outputs_refs_owned: Vec<Vec<&ArrayD<Data>>> =
                outputs_data.iter().map(|vv| vv.iter().collect::<Vec<_>>()).collect();
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
            let bytes = to_vec_bytes(&proofs)?;
            return Ok((PyBytes::new(py, &bytes).into(), None));
         }


        
        #[cfg(feature = "fold")]
        {
            let outputs_refs_owned: Vec<Vec<&ArrayD<Data>>> =
                outputs_data.iter().map(|vv| vv.iter().collect::<Vec<_>>()).collect();
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
            return Ok((PyBytes::new(py, &pbytes).into(), Some(PyBytes::new(py, &abytes).into())));

        }

    }

    /// Verify proofs (and accumulator proofs if built with `fold`).
    pub fn verify(&self, py: Python<'_>, proofs: &PyAny, acc_proofs: Option<&PyAny>) -> PyResult<bool> {

        let mut timing = TimingTree::new("verify", log::Level::Info);
        
        let inputs_enc = self.last_inputs_enc.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("verify() requires prior prove()"))?;
        let outputs_enc = self.last_outputs_enc.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("verify() requires prior prove()"))?;

        // Derive the same FS seed as in `prove()`
        let seed = fs_seed_from_encodings(&self.models_enc, inputs_enc, outputs_enc)?;
        let mut rng = StdRng::from_seed(seed);
        
        // helper to read a Python object into Vec<u8> (safe, no `unsafe`)
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

        // Deserialize inputs for verify
        // For non-fold: Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)>
        // For fold: Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)> + Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>, Vec<_Gt_>)>
        
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
}

#[pymodule]
fn pyzktorch(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<ZkTorchSession>()?;
    Ok(())
}