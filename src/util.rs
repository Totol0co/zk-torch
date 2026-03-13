pub use arithmetic::*;
pub use config::*;
pub use copy_constraint::*;
pub use err::*;
pub use fft::*;
pub use fold::*;
pub use iter::*;
pub use msm::*;
pub use onnx::*;
pub use poly::*;
pub use prover::*;
pub use random::*;
pub use serialization::*;
pub use shape::*;
pub use verifier::*;

pub mod arithmetic;
pub mod config;
pub mod copy_constraint;
pub mod err;
pub mod fft;
pub mod fold;
pub mod iter;
pub mod msm;
pub mod onnx;
pub mod poly;
pub mod prover;
pub mod random;
pub mod serialization;
pub mod shape;
pub mod verifier;

use ndarray::{ArrayD, IxDyn};
use crate::basic_block::{Data, SRS};
use ark_bn254::Fr;
use ndarray::Axis;

/// Convertit un ArrayD<Fr> (N‑D) en ArrayD<Data>
/// mais SANS blinding (r = 0), grâce à Data::new_public().
///
/// Ce convertisseur est destiné *uniquement* aux inputs/outputs publics.
/// Il produit des engagements KZG déterministes, et donc reproductibles
/// par le vérifieur.
pub fn convert_to_data_public(srs: &SRS, arr: &ArrayD<Fr>) -> ArrayD<Data> {
    let nd = arr.ndim();
    assert!(nd >= 1, "convert_to_data_public: need at least 1D tensor");

    let shape = arr.shape();
    let last = shape[nd - 1];
    let outer_count = arr.len() / last;

    // On itère sur les "lanes" (tranches) le long du dernier axe
    // et on construit un Vec<Data> en ordre row-major.
    let mut acc: Vec<Data> = Vec::with_capacity(outer_count);

    for lane in arr.view().lanes(Axis(nd - 1)) {
        // lane: ArrayView1<Fr> (longueur = last)
        // on préfère un slice contigu; sinon on copie
        let slice: Vec<Fr>;
        let raw_slice: &[Fr] = if let Some(s) = lane.as_slice() {
            s
        } else {
            slice = lane.to_owned().into_raw_vec();
            &slice
        };
        acc.push(Data::new_public(srs, raw_slice));
    }

    // Shape de sortie = shape sans le dernier axe
    let out_shape = IxDyn(&shape[..nd - 1]);

    ArrayD::from_shape_vec(out_shape, acc)
        .expect("convert_to_data_public: shape reconstruction failed")
}

