use ark_ec::{VariableBaseMSM, pairing::Pairing};
use ark_poly::{GeneralEvaluationDomain, EvaluationDomain};
use ark_bls12_381::{Fr, G1Projective, G1Affine, G2Projective, G2Affine, Bls12_381};
use ark_std::{ops::Mul, ops::Sub};
use rand::Rng;
use super::{BasicBlock,Data,DataEnc};

pub struct MulBasicBlock;
impl BasicBlock for MulBasicBlock{
  fn run(_: &[Fr],
         inputs: &[&[Fr]]) ->
         Vec<Fr>{
    let mut r = Vec::new();
    for i in 0..inputs[0].len(){
      r.push(inputs[0][i]*inputs[1][i]);
    }
    return r;
  }
  fn setup(_: (&[G1Affine],&[G2Affine]),
           _: &mut Data) ->
          (Vec<G1Affine>,Vec<G2Affine>){
    return (Vec::new(), Vec::new());
  }
  fn prove<R: Rng>(srs: (&[G1Affine],&[G2Affine]),
                   _: (&[G1Affine],&[G2Affine]),
                   _: &Data,
                   inputs: &[&Data],
                   output: &Data,
                   _: &mut R) ->
                  (Vec<G1Affine>,Vec<G2Affine>,Vec<Fr>){
    let N = inputs[0].raw.len();
    let domain  = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let gx2 = G2Projective::msm(&srs.1[..N], &inputs[1].poly.coeffs).unwrap().into();
    let t = inputs[0].poly.mul(&inputs[1].poly).sub(&output.poly).divide_by_vanishing_poly(domain).unwrap().0;
    let tx = G1Projective::msm(&srs.0[..N-1], &t.coeffs).unwrap().into();
    return (vec![tx],vec![gx2],Vec::new());
  }
  fn verify<R: Rng>(srs: (&[G1Affine],&[G2Affine]),
                    _: &DataEnc,
                    inputs: &[&DataEnc],
                    output: &DataEnc,
                    proof: (&[G1Affine],&[G2Affine],&[Fr]),
                    _: &mut R){
    // Verify f(x)*g(x)-h(x)=z(x)t(x)
    let lhs = Bls12_381::pairing(inputs[0].g1,proof.1[0]) - Bls12_381::pairing(output.g1,srs.1[0]);
    let rhs = Bls12_381::pairing(proof.0[0],srs.1[inputs[0].len]-srs.1[0]);
    assert!(lhs==rhs);
    // Verify gx2
    let lhs = Bls12_381::pairing(inputs[1].g1,srs.1[0]);
    let rhs = Bls12_381::pairing(srs.0[0],proof.1[0]);
    assert!(lhs==rhs);
  }
}

