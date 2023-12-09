use ark_ec::{AffineRepr, VariableBaseMSM, pairing::Pairing};
use ark_poly::{evaluations::univariate::Evaluations, GeneralEvaluationDomain, EvaluationDomain};
use ark_bls12_381::{Fr, G1Projective, G1Affine, G2Projective, G2Affine, Bls12_381};
use ark_std::{ops::Mul, ops::Sub, UniformRand};
use std::time::Instant;
const N : usize = 1<<5;
const n : usize = 1<<2;

fn main() {
  let mut rng = ark_std::test_rng();
  let x = Fr::rand(&mut rng);
  let mut xp = x;
  let mut srs = [G1Affine::generator() ; N];
  for i in 1..N{
    srs[i]=(srs[i]*xp).into();
    xp*=x;
  }
  xp = x;
  let mut srs2 = [G2Affine::generator() ; N+1];
  for i in 1..N+1{
    srs2[i]=(srs2[i]*xp).into();
    xp*=x;
  }
  let domain_N  = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
  let domain_n  = GeneralEvaluationDomain::<Fr>::new(n).unwrap();

  let mut table = Vec::new();
  let mut f_i = Vec::new();
  for i in 0..N{
    table.push(Fr::from(i as u32));
  }
  for i in 0..n{
    f_i.push(Fr::from(i as u32));
  }
  let f = Evaluations::from_vec_and_domain(f_i, domain_n).interpolate();
  let Z_V_x_2 = srs2[N] - srs2[0];
  let T = Evaluations::from_vec_and_domain(table, domain_N).interpolate();
  let T_x_2 = G2Projective::msm(&srs2[..N], &T.coeffs).unwrap();
  let f_x = G1Projective::msm(&srs[..n], &f.coeffs).unwrap();

}

