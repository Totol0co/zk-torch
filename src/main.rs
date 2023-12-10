use ark_ec::{Group};
use ark_poly::{GeneralEvaluationDomain, EvaluationDomain};
use ark_bls12_381::{Fr, G1Projective, G1Affine};
use ark_std::{ops::Mul};
mod util;
mod g_fft;

fn main() {
  let domain  = GeneralEvaluationDomain::<Fr>::new(4).unwrap();
  let a = vec![Fr::from(1),Fr::from(2),Fr::from(3)];
  let x = vec![Fr::from(9),Fr::from(2)];
  let z = vec![Fr::from(20),Fr::from(31)];

  let x_g : Vec<G1Projective> = x.iter().map(|y| G1Projective::generator().mul(*y)).collect();
  let res = util::toeplitz_mul(domain, &a, &x_g);
  let res_affine : Vec<G1Affine> = res.iter().map(|y| (*y).into()).collect();
  println!("{:?}",res_affine);

  let z_g : Vec<G1Affine> = z.iter().map(|y| G1Projective::generator().mul(*y).into()).collect();
  println!("{:?}",z_g);

}

