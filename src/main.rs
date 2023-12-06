use ark_ec::AffineRepr;
use ark_ec::VariableBaseMSM;
use ark_ec::pairing::Pairing;
use ark_poly::evaluations::univariate::Evaluations;
use ark_poly::{GeneralEvaluationDomain, EvaluationDomain};
use ark_bls12_381::{Fr, G1Projective, G1Affine, G2Projective, G2Affine, Bls12_381};
use ark_std::{ops::Mul, ops::Sub, UniformRand};
use std::time::Instant;
const N : usize = 1<<10;

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

  let domain  = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
  let mut a = Vec::new();
  let mut b = Vec::new();
  let mut c = Vec::new();
  for _ in 0..N{
    let x = Fr::rand(&mut rng);
    let y = Fr::rand(&mut rng);
    a.push(x);
    b.push(y);
    c.push(x*y);
  }
  let f = Evaluations::from_vec_and_domain(a, domain).interpolate();
  let g = Evaluations::from_vec_and_domain(b, domain).interpolate();
  let h = Evaluations::from_vec_and_domain(c, domain).interpolate();
  let fx = G1Projective::msm(&srs, &f.coeffs).unwrap();
  let gx2 = G2Projective::msm(&srs2[..N], &g.coeffs).unwrap();
  let hx = G1Projective::msm(&srs, &h.coeffs).unwrap();
  let zx2 = srs2[N] - srs2[0];

  let start = Instant::now();
  let t = f.mul(&g).sub(&h).divide_by_vanishing_poly(domain).unwrap().0;
  println!("{:?}", start.elapsed());
  let tx = G1Projective::msm(&srs[..N-1], &t.coeffs).unwrap();
  println!("{:?}", start.elapsed());

  let lhs = Bls12_381::pairing(fx,gx2) - Bls12_381::pairing(hx,srs2[0]);
  let rhs = Bls12_381::pairing(tx,zx2);
  
  assert!(lhs==rhs);
  println!("{:?}", start.elapsed());
}
