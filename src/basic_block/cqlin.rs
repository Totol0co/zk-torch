#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_imports)]
#![allow(unused_variables)]
use ark_ec::{VariableBaseMSM, AffineRepr, pairing::Pairing};
use ark_ff::Field;
use ark_poly::{evaluations::univariate::Evaluations, GeneralEvaluationDomain, EvaluationDomain,univariate::DensePolynomial, Polynomial};
use ark_bls12_381::{Fr, G1Projective, G2Projective, G1Affine, G2Affine, Bls12_381};
use ark_std::{Zero, One, ops::{Mul,Div,Sub}, UniformRand};
use std::collections::HashMap;
use rand::Rng;
use super::{BasicBlock,Data,DataEnc};
use crate::util;

pub struct CQLinBasicBlock;
impl BasicBlock for CQLinBasicBlock{
  fn run(model: &Vec<Fr>,
         inputs: &Vec<Vec<Fr>>) ->
         Vec<Fr>{
    let n = inputs[0].len();
    let mut r = vec![Fr::zero() ; n];
    for i in 0..n{
      for j in 0..n{
        r[i]+=model[i*n+j] * inputs[0][j];
      }
    }
    return r;
  }
  fn setup(srs: (&Vec<G1Affine>,&Vec<G2Affine>),
           model: &Data) ->
          (Vec<G1Affine>,Vec<G2Affine>){
    let N = model.raw.len();
    let n : usize= (N as f64).sqrt() as usize;
    let n_inv = Fr::from(n as u64).inverse().unwrap();
    let domain_2n  = GeneralEvaluationDomain::<Fr>::new(2*n).unwrap();
    let domain_n  = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
    let srs_p : Vec<G1Projective> = srs.0.iter().map(|x| (*x).into()).collect();
    let L_i_x = util::ifft(domain_n, &srs_p[..n]);
    let L_i_x_n = util::ifft(domain_n, &(0..n).map(|i| srs_p[n*i]).collect::<Vec<_>>());

    let mut temp: Vec<Vec<_>>= (0..n).map(|i|(0..n).map(|j|srs_p[i+n*j]).collect()).collect();
    temp.iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));
    let mut U: Vec<Vec<_>> = (0..n).map(|j|(0..n).map(|i|temp[i][j]).collect()).collect();
    U.iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));
    let mut temp: Vec<Vec<G2Projective>> = (0..n).map(|i|(0..n).map(|j|srs.1[i+n*j].into()).collect()).collect();
    temp.iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));
    let mut U2: Vec<Vec<_>> = (0..n).map(|i|(0..n).map(|j|temp[j][i]).collect()).collect();
    U2.iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));
    let mut V = util::ifft(domain_n, &srs_p[N-n..N]);
    V.iter_mut().for_each(|x| *x *= n_inv);

    let mut srs_star: Vec<Vec<_>>= (0..n).map(|i|(0..n).map(|j|srs_p[i+n*j]).collect()).collect();
    srs_star.iter_mut().for_each(|x| util::ifft_in_place(domain_n, x));
    srs_star = (0..n).map(|i|(0..n).map(|j|srs_star[j][n-1-i]).collect()).collect();
    srs_star.iter_mut().for_each(|x| x.append(&mut vec![G1Projective::zero(); n]));
    srs_star.iter_mut().for_each(|x| util::fft_in_place(domain_2n, x));
    
    let mut Ls = vec![vec![Fr::zero() ; n] ; n];
    Ls.iter_mut().enumerate().for_each(|(i, x)| x[i]=Fr::one());
    Ls.iter_mut().for_each(|x| domain_n.ifft_in_place(x));
    let S: Vec<Vec<_>>= (0..n).map(|i|(0..n).map(|j|(U[i][j]*domain_n.element(i).inverse().unwrap()-V[j]) * model.raw[i*n+j]).collect()).collect();
    let S: Vec<_> = S.iter().map(|x|x.iter().sum::<G1Projective>()).collect();
    let R: Vec<Vec<_>>= (0..n).map(|i|(0..n).map(|j|U[i][j] * model.raw[i*n+j]).collect()).collect();
    let R: Vec<_> = R.iter().map(|x|x.iter().sum::<G1Projective>()).collect();
    let C: Vec<Vec<Vec<_>>>= (0..n).map(|i|(0..n).map(|coeff|(0..n).map(|j|model.raw[j*n+i]*Ls[j][coeff]).collect()).collect()).collect();
    let C: Vec<Vec<_>> = C.iter().map(|x|x.iter().map(|y|y.iter().sum::<Fr>()).collect()).collect();

    let mut temp = C;
    temp.iter_mut().for_each(|x| x.append(&mut vec![Fr::zero(); n]));
    temp.iter_mut().for_each(|x| domain_2n.fft_in_place(x));
    let temp: Vec<Vec<_>> = (0..2*n).map(|i|(0..n).map(|j|srs_star[j][i]*temp[j][i]).collect()).collect();
    let mut temp: Vec<_> = temp.iter().map(|x|x.iter().sum::<G1Projective>()).collect();
    util::ifft_in_place(domain_2n, &mut temp);
    let temp = util::fft(domain_n, &temp[n..]);
    let Q: Vec<_> = (0..n).map(|i|temp[i]*domain_n.element(i)*n_inv).collect();
    let M_x = (0..n).map(|i|(0..n).map(|j|U2[i][j]*model.raw[i*n+j]).sum::<G2Projective>()).sum::<G2Projective>();
    
    let R: Vec<G1Affine> = R.iter().map(|x|(*x).into()).collect();
    let mut Q: Vec<G1Affine> = Q.iter().map(|x|(*x).into()).collect();
    let mut S: Vec<G1Affine> = S.iter().map(|x|(*x).into()).collect();
    let mut setup = R;
    setup.append(&mut Q);
    setup.append(&mut S);
    return (setup,vec![M_x.into()]);
  }
  fn prove<R: Rng>(srs: (&Vec<G1Affine>,&Vec<G2Affine>),
                   setup: (&Vec<G1Affine>,&Vec<G2Affine>),
                   model: &Data,
                   inputs: &Vec<Data>,
                   _output: &Data,
                   rng: &mut R) ->
                  (Vec<G1Affine>,Vec<G2Affine>,Vec<Fr>){
    return (Vec::new(),Vec::new(),Vec::new());
  }
  fn verify<R: Rng>(srs: (&Vec<G1Affine>,&Vec<G2Affine>),
                    model: &DataEnc,
                    inputs: &Vec<DataEnc>,
                    _output: &DataEnc,
                    proof: (&Vec<G1Affine>,&Vec<G2Affine>,&Vec<Fr>),
                    rng: &mut R){
  }
}


