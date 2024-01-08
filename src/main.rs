#![allow(non_snake_case)]
use ark_ec::{AffineRepr, VariableBaseMSM, pairing::Pairing};
use ark_poly::{univariate::DensePolynomial, GeneralEvaluationDomain, EvaluationDomain};
use ark_bls12_381::{Fr, G1Projective, G1Affine, G2Projective, G2Affine, Bls12_381};
use ark_std::{ops::Mul, ops::Sub, UniformRand};
use std::time::Instant;
use rand::Rng;

struct Data{
  raw : Vec<Fr>,
  poly: DensePolynomial<Fr>,
  g1: G1Affine
}
struct DataEnc{
  len : usize,
  g1: G1Affine
}
trait BasicBlock{
  fn run(model: &[Fr],
         inputs: &[&[Fr]]) ->
         Vec<Fr>;
  fn setup(srs: (&[G1Affine],&[G2Affine]),
           model: &mut Data) ->
          (Vec<G1Affine>,Vec<G2Affine>);
  fn prove<R: Rng>(srs: (&[G1Affine],&[G2Affine]),
                   setup: (&[G1Affine],&[G2Affine]),
                   model: &Data,
                   inputs: &[&Data],
                   output: &Data,
                   rng: &mut R) ->
                  (Vec<G1Affine>,Vec<G2Affine>);
  fn verify<R: Rng>(srs: (&[G1Affine],&[G2Affine]),
                    model: &DataEnc,
                    inputs: &[&DataEnc],
                    output: &DataEnc,
                    proof: (&[G1Affine],&[G2Affine]),
                    rng: &mut R);
}
struct MulBasicBlock;
impl BasicBlock for MulBasicBlock{
  fn run(model: &[Fr],
         inputs: &[&[Fr]]) ->
         Vec<Fr>{
    let mut r = Vec::new();
    for i in 0..inputs[0].len(){
      r.push(inputs[0][i]*inputs[1][i]);
    }
    return r;
  }
  fn setup(srs: (&[G1Affine],&[G2Affine]),
           model: &mut Data) ->
          (Vec<G1Affine>,Vec<G2Affine>){
    return (Vec::new(), Vec::new());
  }
  fn prove<R: Rng>(srs: (&[G1Affine],&[G2Affine]),
                   setup: (&[G1Affine],&[G2Affine]),
                   model: &Data,
                   inputs: &[&Data],
                   output: &Data,
                   rng: &mut R) ->
                  (Vec<G1Affine>,Vec<G2Affine>){
    let N = inputs[0].raw.len();
    let domain  = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let gx2 = G2Projective::msm(&srs.1[..N], &inputs[1].poly.coeffs).unwrap().into();
    let t = inputs[0].poly.mul(&inputs[1].poly).sub(&output.poly).divide_by_vanishing_poly(domain).unwrap().0;
    let tx = G1Projective::msm(&srs.0[..N-1], &t.coeffs).unwrap().into();
    return (vec![tx],vec![gx2]);
  }
  fn verify<R: Rng>(srs: (&[G1Affine],&[G2Affine]),
                    model: &DataEnc,
                    inputs: &[&DataEnc],
                    output: &DataEnc,
                    proof: (&[G1Affine],&[G2Affine]),
                    rng: &mut R){
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
  let inputs : Vec<&[Fr]> = vec![&a, &b];
  let c = MulBasicBlock::run(&Vec::new(),&inputs);
  let mut model = Data{raw: Vec::new(), poly: DensePolynomial{coeffs:Vec::new()}, g1: G1Affine::generator()};
  let setup = MulBasicBlock::setup(srs,&mut model);
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
  let proof = MulBasicBlock::prove(srs,(&(setup.0),&(setup.1)),&model,&inputs,&c,&mut rng);
  let a = DataEnc{len: a.raw.len(), g1: a.g1};
  let b = DataEnc{len: b.raw.len(), g1: b.g1};
  let c = DataEnc{len: c.raw.len(), g1: c.g1};
  let inputs : Vec<&DataEnc> = vec![&a, &b];
  let model = DataEnc{len: 0, g1: G1Affine::generator()};
  MulBasicBlock::verify(srs,&model,&inputs,&c,(&(proof.0),&(proof.1)),&mut rng);
}
