#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use ark_ec::{VariableBaseMSM, AffineRepr, pairing::Pairing};
use ark_ff::{Field};
use ark_poly::{evaluations::univariate::Evaluations, GeneralEvaluationDomain, EvaluationDomain,univariate::DensePolynomial, Polynomial};
use ark_bls12_381::{Fr, G1Projective, G2Projective, G1Affine, G2Affine, Bls12_381};
use ark_std::{Zero, One, ops::{Mul,Div,Sub}, UniformRand};
use std::collections::HashMap;
use rand::Rng;
use super::{BasicBlock,Data,DataEnc};
use crate::{util,g_fft};

pub struct CQBasicBlock;
impl BasicBlock for CQBasicBlock{
  fn run(_: &[Fr],
         _: &[&[Fr]]) ->
         Vec<Fr>{
    return Vec::new();
  }
  fn setup(srs: (&[G1Affine],&[G2Affine]),
           model: &mut Data) ->
          (Vec<G1Affine>,Vec<G2Affine>){
    let N = model.raw.len();
    let domain_2N  = GeneralEvaluationDomain::<Fr>::new(2*N).unwrap();
    let domain_N  = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let srs_p : Vec<G1Projective> = srs.0.iter().map(|x| (*x).into()).collect();
    let T_x_2 = G2Projective::msm(&srs.1[..N], &model.poly.coeffs).unwrap().into();
    let mut temp = model.poly.coeffs[1..].to_vec();
    temp.resize(N*2-1,Fr::zero());
    let mut temp2 = srs_p.clone();
    temp2.reverse();
    let mut Q_i_x_1 = util::toeplitz_mul(domain_2N, &temp, &temp2);
    g_fft::fft_in_place(domain_N, &mut Q_i_x_1);
    let temp = Fr::from(N as u32).inverse().unwrap();
    let temp2 = domain_N.group_gen_inv().pow(&[(N-1) as u64]);
    for i in 0..N{
      Q_i_x_1[i] *= temp * temp2.pow(&[i as u64]);
    }
    let L_i_x_1 = g_fft::ifft(domain_N, &srs_p[..N]);
    let mut L_i_0_x_1 = L_i_x_1.clone();
    let temp = srs.0[N-1] * Fr::from(N as u64).inverse().unwrap();
    for i in 0..N{
      L_i_0_x_1[i] *= domain_N.group_gen_inv().pow(&[i as u64]);
      L_i_0_x_1[i] -= temp;
    }
    let Q_i_x_1 : Vec<G1Affine> = Q_i_x_1.iter().map(|x| (*x).into()).collect();
    let L_i_x_1 : Vec<G1Affine> = L_i_x_1.iter().map(|x| (*x).into()).collect();
    let L_i_0_x_1 : Vec<G1Affine> = L_i_0_x_1.iter().map(|x| (*x).into()).collect();
    let mut setup = Q_i_x_1;
    setup.extend(L_i_x_1);
    setup.extend(L_i_0_x_1);
    return (setup,vec![T_x_2]);
  }
  fn prove<R: Rng>(srs: (&[G1Affine],&[G2Affine]),
                   setup: (&[G1Affine],&[G2Affine]),
                   model: &Data,
                   inputs: &[&Data],
                   _: &Data,
                   rng: &mut R) ->
                  (Vec<G1Affine>,Vec<G2Affine>,Vec<Fr>){
    let N = model.raw.len();
    let n = inputs[0].raw.len();
    let domain_n  = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    // gen(N, t):
    let Q_i_x_1 = &setup.0[..N];
    let L_i_x_1 = &setup.0[N..2*N];
    let L_i_0_x_1 = &setup.0[2*N..];
    let mut table_dict = HashMap::new();
    for i in 0..N{
      table_dict.insert(model.raw[i],i);
    }
    
    // Round 1
    let mut m_i = HashMap::new();
    for i in 0..n{
      m_i.entry(table_dict.get(&inputs[0].raw[i]).unwrap()).and_modify(|x| *x+=1).or_insert(1);
    }
    let (temp, temp2) : (Vec<G1Affine>,Vec<Fr>)=m_i.iter().map(|(i,y)| (L_i_x_1[**i], Fr::from(*y as u32))).unzip();
    let m_x_1 = G1Projective::msm(&temp, &temp2).unwrap().into();

    //Round 2
    let beta = Fr::rand(rng);
    let A_i : HashMap<usize,Fr>= m_i.iter().map(|(i,y)| (**i,Fr::from(*y as u32) * (model.raw[**i]+beta).inverse().unwrap())).collect();
    let (temp, temp2) : (Vec<G1Affine>,Vec<Fr>)=A_i.iter().map(|(i,y)| (L_i_x_1[*i], *y)).unzip();
    let A_x_1 = G1Projective::msm(&temp, &temp2).unwrap().into();
    let (temp, temp2) : (Vec<G1Affine>,Vec<Fr>)=A_i.iter().map(|(i,y)| (Q_i_x_1[*i], *y)).unzip();
    let Q_A_x_1 = G1Projective::msm(&temp, &temp2).unwrap().into();
    let B_i : Vec<Fr>= (0..n).map(|i| (inputs[0].raw[i]+beta).inverse().unwrap()).collect();
    let B = Evaluations::from_vec_and_domain(B_i.clone(), domain_n).interpolate();
    let B_0 = DensePolynomial{coeffs : B.coeffs[1..].to_vec()};
    let B_0_x_1 = G1Projective::msm_unchecked(&srs.0[0..n-1], &B_0).into();
    let mut Q_B = B.mul(&(inputs[0].poly.clone() + (DensePolynomial{coeffs:vec![beta]})));
    Q_B = Q_B.sub(&DensePolynomial{coeffs:vec![Fr::one()]}).divide_by_vanishing_poly(domain_n).unwrap().0;
    let Q_B_x_1 = G1Projective::msm_unchecked(&srs.0[0..n-1], &Q_B).into();
    let P_x_1 = G1Projective::msm_unchecked(&srs.0[N-n+1..N], &B_0).into();

    //Round 3
    let gamma = Fr::rand(rng);
    let eta = Fr::rand(rng);
    let B_0_gamma = B_0.evaluate(&gamma);
    let f_gamma = inputs[0].poly.evaluate(&gamma);
    let A_0 = Fr::from(N as u32).inverse().unwrap() * A_i.iter().map(|(_,y)| *y).sum::<Fr>();
    let b_0 = Fr::from(N as u32) * A_0 * Fr::from(n as u32).inverse().unwrap();
    let Z_H_gamma = domain_n.evaluate_vanishing_polynomial(gamma);
    let b_gamma = B_0_gamma * gamma + b_0;
    let Q_b_gamma = (b_gamma * (f_gamma + beta) - Fr::one()) * Z_H_gamma.inverse().unwrap();
    let v = B_0_gamma + eta * f_gamma + eta * eta * Q_b_gamma;

    let mut num = B_0 + inputs[0].poly.mul(eta)+ Q_B.mul(eta * eta);
    num -= &DensePolynomial{coeffs:vec![v]};
    let h = num.div(&DensePolynomial{coeffs:vec![-gamma,Fr::one()]});
    let pi_gamma = G1Projective::msm_unchecked(&srs.0[..n-1],&h).into();

    //TESTING
    let c = B_0_x_1 + inputs[0].g1 * eta + Q_B_x_1 * eta * eta;
    let lhs = Bls12_381::pairing(c - G1Affine::generator() * v + pi_gamma * gamma, srs.1[0]);
    let rhs = Bls12_381::pairing(pi_gamma, srs.1[1]);
    assert!(lhs==rhs);
    //TESTING

    let (temp, temp2) : (Vec<G1Affine>,Vec<Fr>)=A_i.iter().map(|(i,y)| (L_i_0_x_1[*i], *y)).unzip();
    let A_0_x = G1Projective::msm(&temp, &temp2).unwrap().into();//11
    return (vec![m_x_1,A_x_1,Q_A_x_1,B_0_x_1,Q_B_x_1,P_x_1,pi_gamma,A_0_x],vec![setup.1[0]],vec![B_0_gamma,f_gamma,A_0]);
  }
  fn verify<R: Rng>(srs: (&[G1Affine],&[G2Affine]),
                    model: &DataEnc,
                    inputs: &[&DataEnc],
                    _: &DataEnc,
                    proof: (&[G1Affine],&[G2Affine],&[Fr]),
                    rng: &mut R){
    let N = model.len;
    let n = inputs[0].len;
    let domain_n  = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let [m_x_1,A_x_1,Q_A_x_1,B_0_x_1,Q_B_x_1,P_x_1,pi_gamma,A_0_x] = proof.0.to_vec()[..] else{panic!("wrong proof format")};
    let [T_x_2] = proof.1.to_vec()[..] else{panic!("wrong proof format")};
    let [B_0_gamma,f_gamma,A_0] = proof.2.to_vec()[..] else{panic!("wrong proof format")};

    // Round 2
    let beta = Fr::rand(rng);
    let A_x_1 : G1Projective = A_x_1.into();
    let m_x_1 : G1Projective = m_x_1.into();
    let lhs = Bls12_381::pairing(A_x_1,T_x_2);
    let rhs = Bls12_381::pairing(Q_A_x_1,srs.1[N] - srs.1[0]) + Bls12_381::pairing(m_x_1 - A_x_1 * beta, srs.1[0]);
    assert!(lhs==rhs);
    let lhs = Bls12_381::pairing(B_0_x_1,srs.1[N-1-(n-2)]);
    let rhs = Bls12_381::pairing(P_x_1,srs.1[0]);
    assert!(lhs==rhs);

    // Round 3
    let gamma = Fr::rand(rng);
    let eta = Fr::rand(rng);
    let b_0 = Fr::from(N as u32) * A_0 * Fr::from(n as u32).inverse().unwrap(); // V computes
    let Z_H_gamma = domain_n.evaluate_vanishing_polynomial(gamma);
    let b_gamma = B_0_gamma * gamma + b_0;
    let Q_b_gamma = (b_gamma * (f_gamma + beta) - Fr::one()) * Z_H_gamma.inverse().unwrap();
    let v = B_0_gamma + eta * f_gamma + eta * eta * Q_b_gamma; // P and V compute
    let c = B_0_x_1 + inputs[0].g1 * eta + Q_B_x_1 * eta * eta; // V computes
    let lhs = Bls12_381::pairing(c - G1Affine::generator() * v + pi_gamma * gamma, srs.1[0]);
    let rhs = Bls12_381::pairing(pi_gamma, srs.1[1]);
    assert!(lhs==rhs);
    let lhs = Bls12_381::pairing(A_x_1 -  G1Affine::generator() * A_0, srs.1[0]);
    let rhs = Bls12_381::pairing(A_0_x, srs.1[1]);
    assert!(lhs==rhs);
  }
}


