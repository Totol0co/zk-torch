use ark_poly::{GeneralEvaluationDomain, EvaluationDomain};
use ark_bls12_381::{Fr, G1Projective};
use ark_std::{Zero};
use crate::g_fft;

pub fn circulant_mul(domain: GeneralEvaluationDomain<Fr>, c: &[Fr], a: &[G1Projective]) -> Vec<G1Projective>{
  let lambda = domain.fft(c);
  let mut r = g_fft::fft(domain,a);
  for i in 0..r.len(){
    r[i] *= lambda[i];
  }
  g_fft::ifft_in_place(domain,&mut r);
  r
}

pub fn toeplitz_mul(domain: GeneralEvaluationDomain<Fr>, m: &[Fr], a: &[G1Projective]) -> Vec<G1Projective>{
  let n = (m.len()+1)/2;
  let mut temp = m.to_vec();
  let mut m2 = temp.split_off(n-1);
  m2.push(Fr::zero());
  m2.append(&mut temp);
  let mut temp2 = a.to_vec();
  temp2.resize(2*n, G1Projective::zero());
  let mut r = circulant_mul(domain, &m2, &temp2);
  r.resize(n, G1Projective::zero());
  r
}
