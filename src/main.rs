use ark_ec::{Group, VariableBaseMSM};
use ark_ff::{Field};
use ark_poly::{evaluations::univariate::Evaluations, GeneralEvaluationDomain, EvaluationDomain};
use ark_bls12_381::{Fr, G1Projective, G2Projective, G1Affine, G2Affine};
use ark_std::{Zero, UniformRand};
use std::time::Instant;
mod util;
mod g_fft;
const N : usize = 1<<5;
const n : usize = 1<<3;

fn main() {
  let mut rng = ark_std::test_rng();
  // gen(N, t):
  let x = Fr::from(5);//Fr::rand(&mut rng);
  let mut xp = x;
  let mut srs = vec![G1Projective::generator() ; N];
  for i in 1..N{
    srs[i]*=xp;
    xp*=x;
  }
  let srs_affine : Vec<G1Affine> = srs.iter().map(|x| (*x).into()).collect();
  xp = x;
  let mut srs2 = vec![G2Projective::generator() ; N+1];
  for i in 1..N+1{
    srs2[i]*=xp;
    xp*=x;
  }
  let srs2_affine : Vec<G2Affine> = srs2.iter().map(|x| (*x).into()).collect();

  let domain_2N  = GeneralEvaluationDomain::<Fr>::new(2*N).unwrap();
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
  let T_x_2 = G2Projective::msm(&srs2_affine[..N], &T.coeffs).unwrap();
  let mut temp = T.coeffs[1..].to_vec();
  temp.resize(N*2-1,Fr::zero());
  let mut temp2 = srs.clone();
  temp2.reverse();
  let mut Q_i_x_1 = util::toeplitz_mul(domain_2N, &temp, &temp2); //K_i_x_1
  g_fft::fft_in_place(domain_N, &mut Q_i_x_1);
  let temp = Fr::from(N as u32).inverse().unwrap();
  let temp2 = domain_N.group_gen_inv().pow(&[(N-1) as u64]);
  for i in 0..N{
    Q_i_x_1[i] *= temp * temp2.pow(&[i as u64]);
  }
  let L_i_x_1 = g_fft::ifft(domain_N, &srs);
  let mut L_i_0_x_1 = L_i_x_1.clone();
  let temp = srs[N-1] * Fr::from(N as u64).inverse().unwrap();
  for i in 0..N{
    L_i_0_x_1[i] *= domain_N.element(i).inverse().unwrap();
    L_i_0_x_1[i] -= temp;
  }
  let f_x = G1Projective::msm(&srs_affine[..n], &f.coeffs).unwrap();
  
}





