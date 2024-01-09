#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use ark_ec::{Group, AffineRepr, VariableBaseMSM, pairing::Pairing};
use ark_ff::{Field};
use ark_poly::{evaluations::univariate::Evaluations, GeneralEvaluationDomain, EvaluationDomain,univariate::DensePolynomial, Polynomial};
use ark_bls12_381::{Fr, G1Projective, G2Projective, G1Affine, G2Affine, Bls12_381};
use ark_std::{Zero, One, ops::{Mul,Div,Sub}, UniformRand};
use std::collections::HashMap;
use std::time::Instant;
const N : usize = 1<<7;
const n : usize = 1<<5;

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

  let mut start = Instant::now();
  let table : Vec<Fr>= (0..N).map(|x| Fr::from(x as u32)).collect();
  let f_i : Vec<Fr>= (0..n).map(|x| Fr::from(x as u32)).collect();
  let f = Evaluations::from_vec_and_domain(f_i.clone(), domain_n).interpolate();
  let Z_V_x_2 = srs2[N] - srs2[0];
  let T = Evaluations::from_vec_and_domain(table.clone(), domain_N).interpolate();
  let T_x_2 = G2Projective::msm(&srs2_affine[..N], &T).unwrap();
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
  let f_x_1 = G1Projective::msm_unchecked(&srs_affine[..n], &f);
  let mut table_dict = HashMap::new();
  for i in 0..N{
    table_dict.insert(table[i],i);
  }
  let Q_i_x_1 : Vec<G1Affine> = Q_i_x_1.iter().map(|x| (*x).into()).collect();
  let L_i_x_1 : Vec<G1Affine> = L_i_x_1.iter().map(|x| (*x).into()).collect();
  let L_i_0_x_1 : Vec<G1Affine> = L_i_0_x_1.iter().map(|x| (*x).into()).collect();
  println!("setup: {:?}",start.elapsed());
  start = Instant::now();
  
  // IsInTable(f_x_1, table, srs; f):
  // Round 1
  let mut m_i = HashMap::new();
  for i in 0..n{
    m_i.entry(table_dict.get(&f_i[i]).unwrap()).and_modify(|x| *x+=1).or_insert(1);
  }
  let (temp, temp2) : (Vec<G1Affine>,Vec<Fr>)=m_i.iter().map(|(i,y)| (L_i_x_1[**i], Fr::from(*y as u32))).unzip();
  let m_x_1 = G1Projective::msm(&temp, &temp2).unwrap();

  //Round 2
  let beta = Fr::rand(&mut rng);
  let A_i : HashMap<usize,Fr>= m_i.iter().map(|(i,y)| (**i,Fr::from(*y as u32) * (table[**i]+beta).inverse().unwrap())).collect();
  let (temp, temp2) : (Vec<G1Affine>,Vec<Fr>)=A_i.iter().map(|(i,y)| (L_i_x_1[*i], *y)).unzip();
  let A_x_1 = G1Projective::msm(&temp, &temp2).unwrap();
  let (temp, temp2) : (Vec<G1Affine>,Vec<Fr>)=A_i.iter().map(|(i,y)| (Q_i_x_1[*i], *y)).unzip();
  let Q_A_x_1 = G1Projective::msm(&temp, &temp2).unwrap();
  let B_i : Vec<Fr>= (0..n).map(|i| (f_i[i]+beta).inverse().unwrap()).collect();
  let B = Evaluations::from_vec_and_domain(B_i.clone(), domain_n).interpolate();
  let B_0 = DensePolynomial{coeffs : B.coeffs[1..].to_vec()};
  let B_0_x_1 = G1Projective::msm_unchecked(&srs_affine[0..n-1], &B_0);
  let mut Q_B = B.mul(&(f.clone() + (DensePolynomial{coeffs:vec![beta]})));
  Q_B = Q_B.sub(&DensePolynomial{coeffs:vec![Fr::one()]}).divide_by_vanishing_poly(domain_n).unwrap().0;
  let Q_B_x_1 = G1Projective::msm_unchecked(&srs_affine[0..n-1], &Q_B);
  let P_x_1 = G1Projective::msm_unchecked(&srs_affine[N-n+1..N], &B_0);
  let lhs = Bls12_381::pairing(A_x_1,T_x_2);
  let rhs = Bls12_381::pairing(Q_A_x_1,Z_V_x_2) + Bls12_381::pairing(m_x_1 - A_x_1 * beta, srs2[0]);
  assert!(lhs==rhs);
  let lhs = Bls12_381::pairing(B_0_x_1,srs2[N-1-(n-2)]);
  let rhs = Bls12_381::pairing(P_x_1,srs2[0]);
  assert!(lhs==rhs);

  //Round 3
  let gamma = Fr::rand(&mut rng);
  let eta = Fr::rand(&mut rng);
  let B_0_gamma = B_0.evaluate(&gamma);
  let f_gamma = f.evaluate(&gamma);
  let A_0 = Fr::from(N as u32).inverse().unwrap() * A_i.iter().map(|(_,y)| *y).sum::<Fr>();
  let b_0 = Fr::from(N as u32) * A_0 * Fr::from(n as u32).inverse().unwrap(); // V computes
  let Z_H_gamma = domain_n.evaluate_vanishing_polynomial(gamma);
  let b_gamma = B_0_gamma * gamma + b_0;
  let Q_b_gamma = (b_gamma * (f_gamma + beta) - Fr::one()) * Z_H_gamma.inverse().unwrap();
  let v = B_0_gamma + eta * f_gamma + eta * eta * Q_b_gamma; // P and V compute
  let mut num = B_0 + f.mul(eta)+ Q_B.mul(eta * eta);
  num -= &DensePolynomial{coeffs:vec![v]};
  let h = num.div(&DensePolynomial{coeffs:vec![-gamma,Fr::one()]});
  let pi_gamma = G1Projective::msm_unchecked(&srs_affine[..n-1],&h);
  let c = B_0_x_1 + f_x_1 * eta + Q_B_x_1 * eta * eta; // V computes
  let lhs = Bls12_381::pairing(c - G1Affine::generator() * v + pi_gamma * gamma, srs2[0]);
  let rhs = Bls12_381::pairing(pi_gamma, srs2[1]);
  assert!(lhs==rhs);
  let (temp, temp2) : (Vec<G1Affine>,Vec<Fr>)=A_i.iter().map(|(i,y)| (L_i_0_x_1[*i], *y)).unzip();
  let A_0_x = G1Projective::msm(&temp, &temp2).unwrap();
  let lhs = Bls12_381::pairing(A_x_1 -  G1Affine::generator() * A_0, srs2[0]);
  let rhs = Bls12_381::pairing(A_0_x, srs2[1]);
  assert!(lhs==rhs);
  println!("proof: {:?}",start.elapsed());
}
