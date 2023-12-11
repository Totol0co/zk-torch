#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use ark_ec::{Group, CurveGroup, VariableBaseMSM, pairing::Pairing};
use ark_ff::{Field};
use ark_poly::{evaluations::univariate::Evaluations, GeneralEvaluationDomain, EvaluationDomain,univariate::DensePolynomial};
use ark_bls12_381::{Fr, G1Projective, G2Projective, G1Affine, G2Affine, Bls12_381};
use ark_std::{Zero, One, ops::{Mul,Add,Sub}, UniformRand};
use std::collections::HashMap;
use std::time::Instant;
mod util;
mod g_fft;
const N : usize = 1<<5;
const n : usize = 1<<3;

fn main() {
  let mut rng = ark_std::test_rng();
  // gen(N, t):
  let x = Fr::rand(&mut rng);
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

  let table : Vec<Fr>= (0..N).map(|x| Fr::from(x as u32)).collect();
  let f_i : Vec<Fr>= (0..n).map(|x| Fr::from(x as u32)).collect();
  let f = Evaluations::from_vec_and_domain(f_i.clone(), domain_n).interpolate();
  let Z_V_x_2 = srs2[N] - srs2[0];
  let T = Evaluations::from_vec_and_domain(table.clone(), domain_N).interpolate();
  let T_x_2 = G2Projective::msm(&srs2_affine[..N], &T.coeffs).unwrap();
  let mut temp = T.coeffs[1..].to_vec();
  temp.resize(N*2-1,Fr::zero());
  let mut temp2 = srs.clone();
  temp2.reverse();
  let mut Q_i_x_1 = util::toeplitz_mul(domain_2N, &temp, &temp2);
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
    L_i_0_x_1[i] *= domain_N.group_gen_inv().pow(&[i as u64]);
    L_i_0_x_1[i] -= temp;
  }
  let f_x_1 = G1Projective::msm(&srs_affine[..n], &f.coeffs).unwrap();
  let mut table_dict = HashMap::new();
  for i in 0..N{
    table_dict.insert(table[i],i);
  }
  
  // IsInTable(f_x_1, table, srs; f):
  // Round 1
  let mut m_i = HashMap::new();
  for i in 0..n{
    m_i.entry(table_dict.get(&f_i[i]).unwrap()).and_modify(|x| *x+=1).or_insert(1);
  }
  let (temp, temp2) : (Vec<G1Affine>,Vec<Fr>)=m_i.iter().map(|(i,y)| (L_i_x_1[**i].into_affine(), Fr::from(*y as u32))).unzip();
  let m_x_1 = G1Projective::msm(&temp, &temp2).unwrap();

  //Round 2
  let beta = Fr::rand(&mut rng);
  let A_i : HashMap<usize,Fr>= m_i.iter().map(|(i,y)| (**i,Fr::from(*y as u32) * (table[**i]+beta).inverse().unwrap())).collect();
  let (temp, temp2) : (Vec<G1Affine>,Vec<Fr>)=A_i.iter().map(|(i,y)| (L_i_x_1[*i].into_affine(), *y)).unzip();
  let A_x_1 = G1Projective::msm(&temp, &temp2).unwrap();
  let (temp, temp2) : (Vec<G1Affine>,Vec<Fr>)=A_i.iter().map(|(i,y)| (Q_i_x_1[*i].into_affine(), *y)).unzip();
  let Q_A_x_1 = G1Projective::msm(&temp, &temp2).unwrap();
  let B_i : Vec<Fr>= (0..n).map(|i| (f_i[i]+beta).inverse().unwrap()).collect();
  let B = Evaluations::from_vec_and_domain(B_i.clone(), domain_n).interpolate();
  let B_0 = DensePolynomial{coeffs : B.coeffs[1..].to_vec()};
  let B_0_x_1 = G1Projective::msm(&srs_affine[0..n-1], &B_0.coeffs).unwrap();
  let mut Q_B = B.mul(&(f.add(DensePolynomial{coeffs:vec![beta]})));
  Q_B = Q_B.sub(&DensePolynomial{coeffs:vec![Fr::one()]}).divide_by_vanishing_poly(domain_n).unwrap().0;
  let Q_B_x_1 = G1Projective::msm(&srs_affine[0..n-1], &Q_B.coeffs).unwrap();
  let P_x_1 = G1Projective::msm(&srs_affine[N-n+1..N], &B_0.coeffs).unwrap();
  let lhs = Bls12_381::pairing(A_x_1,T_x_2);
  let rhs = Bls12_381::pairing(Q_A_x_1,Z_V_x_2) + Bls12_381::pairing(m_x_1 - A_x_1 * beta, srs2[0]);
  assert!(lhs==rhs);
  let lhs = Bls12_381::pairing(B_0_x_1,srs2[N-1-(n-2)]);
  let rhs = Bls12_381::pairing(P_x_1,srs2[0]);
  assert!(lhs==rhs);
}





