#![allow(non_snake_case)]
use ark_ec::{AffineRepr, VariableBaseMSM};
use ark_poly::{univariate::DensePolynomial, GeneralEvaluationDomain, EvaluationDomain};
use ark_bls12_381::{Fr, G1Projective, G1Affine, G2Affine};
use ark_std::{UniformRand};
use std::time::Instant;
use basic_block::*;
mod basic_block;
mod util;
mod g_fft;

fn test<BB: BasicBlock>(){
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
  let inputs : Vec<&[Fr]> = vec![&a, &b];
  let c = BB::run(&Vec::new(),&inputs);
  let mut model = Data{raw: Vec::new(), poly: DensePolynomial{coeffs:Vec::new()}, g1: G1Affine::generator()};
  let setup = BB::setup(srs,&mut model);
  let f = DensePolynomial{coeffs: domain.ifft(&a)};
  let g = DensePolynomial{coeffs: domain.ifft(&b)};
  let h = DensePolynomial{coeffs: domain.ifft(&c)};
  let fx : G1Affine = G1Projective::msm(&srs.0[..N], &f.coeffs).unwrap().into();
  let gx : G1Affine = G1Projective::msm(&srs.0[..N], &g.coeffs).unwrap().into();
  let hx : G1Affine = G1Projective::msm(&srs.0[..N], &h.coeffs).unwrap().into();
  let a = Data{raw: a, poly: f, g1: fx};
  let b = Data{raw: b, poly: g, g1: gx};
  let c = Data{raw: c, poly: h, g1: hx};
  let inputs : Vec<&Data> = vec![&a, &b];
  let proof = BB::prove(srs,(&(setup.0),&(setup.1)),&model,&inputs,&c,&mut rng);
  let a = DataEnc{len: a.raw.len(), g1: a.g1};
  let b = DataEnc{len: b.raw.len(), g1: b.g1};
  let c = DataEnc{len: c.raw.len(), g1: c.g1};
  let inputs : Vec<&DataEnc> = vec![&a, &b];
  let model = DataEnc{len: 0, g1: G1Affine::generator()};
  BB::verify(srs,&model,&inputs,&c,(&(proof.0),&(proof.1)),&mut rng);
}
fn main() {
  test::<MulBasicBlock>();
}
