use ark_ec::Group;
use ark_poly::evaluations::univariate::Evaluations;
use ark_poly::{Polynomial, GeneralEvaluationDomain, EvaluationDomain};
use ark_msm::msm::{VariableBaseMSM};
use ark_bls12_381::{Fr, G1Projective, G1Affine, G2Projective, G2Affine};
use ark_std::{ops::Mul, ops::Sub, UniformRand};
use std::time::Instant;
const N : usize = 1<<14;

fn main() {
  let mut rng = ark_std::test_rng();

  let x = Fr::rand(&mut rng);

  let mut srs_x = [x; N];
  let mut x2 = x;
  for i in 0..N{
    srs_x[i]=x2;
    x2*=x2;
  }
  let bases = [G1Projective::generator().into() ; N];
  let srs = VariableBaseMSM::multi_scalar_mul(&bases, &srs_x);

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

  let start = Instant::now();
  let t = f.mul(&g).sub(&h).divide_by_vanishing_poly(domain).unwrap().0;
  let gamma = Fr::rand(&mut rng);
  let lhs = f.evaluate(&gamma) * g.evaluate(&gamma) - h.evaluate(&gamma);
  let rhs = t.evaluate(&gamma) * domain.vanishing_polynomial().evaluate(&gamma);
  assert!(lhs==rhs);
  println!("{:?}", start.elapsed());
}

