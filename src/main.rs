#![allow(non_snake_case)]
use ark_ec::{AffineRepr, VariableBaseMSM, pairing::Pairing};
use ark_poly::{univariate::DensePolynomial, GeneralEvaluationDomain, EvaluationDomain};
use ark_bls12_381::{Fr, G1Projective, G1Affine, G2Projective, G2Affine, Bls12_381};
use ark_std::{ops::Mul, ops::Sub, UniformRand};
use std::time::Instant;
use rand::Rng;


trait BasicBlock{
  fn setup(srs: (&[G1Affine],&[G2Affine]),
           model: &[Fr]) ->
          (Vec<G1Affine>,Vec<G2Affine>);
  fn prove<R: Rng>(srs: (&[G1Affine],&[G2Affine]),
                   setup: (&[G1Affine],&[G2Affine]),
                   inputs: &[&[Fr]],
                   rng: &mut R) ->
                  ((Vec<G1Affine>,Vec<G2Affine>), Vec<Fr>);
  fn verify<R: Rng>(srs: (&[G1Affine],&[G2Affine]),
                    inputs: &[G1Affine],
                    proof: (&[G1Affine],&[G2Affine]),
                    output: G1Affine,
                    rng: &mut R);
}

struct MulBasicBlock;
impl BasicBlock for MulBasicBlock{
  fn setup(_: (&[G1Affine],&[G2Affine]),
           _: &[Fr]) ->
          (Vec<G1Affine>,Vec<G2Affine>){
    return (Vec::new(), Vec::new());
  }
  fn prove<R: Rng>(srs: (&[G1Affine],&[G2Affine]),
                   _: (&[G1Affine],&[G2Affine]),
                   inputs: &[&[Fr]],
                   _: &mut R) ->
                  ((Vec<G1Affine>,Vec<G2Affine>), Vec<Fr>){
    let N = 1<<3; //TODO: fix hardcode
    let domain  = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let mut r = Vec::new();
    for i in 0..N{
      r.push(inputs[0][i]*inputs[1][i]);
    }
    let f = DensePolynomial{coeffs: domain.ifft(&inputs[0])};
    let g = DensePolynomial{coeffs: domain.ifft(&inputs[1])};
    let h = DensePolynomial{coeffs: domain.ifft(&r)};
    let gx2 = G2Projective::msm(&srs.1[..N], &g.coeffs).unwrap().into();
    let t = f.mul(&g).sub(&h).divide_by_vanishing_poly(domain).unwrap().0;
    let tx = G1Projective::msm(&srs.0[..N-1], &t.coeffs).unwrap().into();
    return ((vec![tx],vec![gx2]),r);
  }
  fn verify<R: Rng>(srs: (&[G1Affine],&[G2Affine]),
                    inputs: &[G1Affine],
                    proof: (&[G1Affine],&[G2Affine]),
                    output: G1Affine,
                    _: &mut R){
    let N = 1<<3; //TODO: fix hardcode
    // Verify f(x)*g(x)-h(x)=z(x)t(x)
    let lhs = Bls12_381::pairing(inputs[0],proof.1[0]) - Bls12_381::pairing(output,srs.1[0]);
    let rhs = Bls12_381::pairing(proof.0[0],srs.1[N]-srs.1[0]);
    assert!(lhs==rhs);
    // Verify gx2
    let lhs = Bls12_381::pairing(inputs[1],srs.1[0]);
    let rhs = Bls12_381::pairing(srs.0[0],proof.1[0]);
    assert!(lhs==rhs);
  }
}

fn main() {
  let mut rng = ark_std::test_rng();
  let x = Fr::rand(&mut rng);
  let mut xp = x;
  const N:usize = 1<<3;
  let domain  = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
  let mut srs = ([G1Affine::generator() ; N], [G2Affine::generator() ; N+1]);
  for i in 1..N{
    srs.0[i]=(srs.0[i]*xp).into();
    xp*=x;
  }
  xp = x;
  for i in 1..N+1{
    srs.1[i]=(srs.1[i]*xp).into();
    xp*=x;
  }
  let srs: (&[G1Affine],&[G2Affine]) = (&(srs.0),&(srs.1));
  let mut a = Vec::new();
  let mut b = Vec::new();
  for _ in 0..N{
    a.push(Fr::rand(&mut rng));
    b.push(Fr::rand(&mut rng));
  }
  let inputs: Vec<&[Fr]> = vec![&a, &b];
  let setup = MulBasicBlock::setup(srs,& Vec::new());
  let (proof,output) = MulBasicBlock::prove(srs,(&(setup.0),&(setup.1)),&inputs,&mut rng);
  let fx : G1Affine = G1Projective::msm(&srs.0[..N], &domain.ifft(&a)).unwrap().into();
  let gx : G1Affine = G1Projective::msm(&srs.0[..N], &domain.ifft(&b)).unwrap().into();
  let hx : G1Affine = G1Projective::msm(&srs.0[..N], &domain.ifft(&output)).unwrap().into();
  let inputs: Vec<G1Affine> = vec![fx, gx];
  MulBasicBlock::verify(srs,&inputs,(&(proof.0),&(proof.1)),hx,&mut rng);
}
