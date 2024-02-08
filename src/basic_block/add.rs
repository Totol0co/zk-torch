use super::{BasicBlock, Data, DataEnc};
use ark_bn254::{Bn254, Fr, G1Affine, G2Affine};
use ark_ec::pairing::Pairing;
use rand::Rng;

pub struct AddBasicBlock;
impl BasicBlock for AddBasicBlock {
  fn run(_model: &Vec<Fr>, inputs: &Vec<Vec<Fr>>) -> Vec<Fr> {
    let mut r = Vec::new();
    for i in 0..inputs[0].len() {
      r.push(inputs[0][i] + inputs[1][i]);
    }
    return r;
  }
  fn prove<R: Rng>(
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &Data,
    inputs: &Vec<Data>,
    output: &Data,
    rng: &mut R,
  ) -> (Vec<G1Affine>, Vec<G2Affine>) {
    // Blinding
    let C = srs.0[0] * (inputs[0].r + inputs[1].r - output.r);
    (vec![C.into()], Vec::new())
  }
  fn verify<R: Rng>(
    srs: (&Vec<G1Affine>, &Vec<G2Affine>),
    _model: &DataEnc,
    inputs: &Vec<DataEnc>,
    output: &DataEnc,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>),
    _rng: &mut R,
  ) {
    // Verify f(x)+g(x)=h(x)
    let lhs = Bn254::pairing(inputs[0].g1 + inputs[1].g1, srs.1[0]);
    let rhs = Bn254::pairing(output.g1, srs.1[0]) + Bn254::pairing(proof.0[0], srs.1[srs.1.len() - 1]);
    assert!(lhs == rhs);
  }
}
