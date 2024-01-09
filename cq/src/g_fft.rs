use ark_poly::{GeneralEvaluationDomain, EvaluationDomain};
use ark_bls12_381::{Fr, G1Projective};
use ark_ff::fields::{Field};
use ark_std::{One};

fn bitreverse(mut n: u32, l: u64) -> u32 {
  let mut r = 0;
  for _ in 0..l {
    r = (r << 1) | (n & 1);
    n >>= 1;
  }
  r
}
pub fn fft_helper(a: &mut [G1Projective], omega: Fr, log_size: u64) {
  let n = a.len();
  let mut m = 1;
  for k in 0..n {
    let rk = bitreverse(k as u32, log_size) as usize;
    if k < rk {
      a.swap(k, rk);
    }
  }
  for _ in 0..log_size {
    let w_m = omega.pow([(n / (2 * m)) as u64]);
    let mut k = 0;
    while k < n {
      let mut w = Fr::one();
      for j in 0..m {
        let mut t = a[(k + m) + j];
        t  *= w;
        a[(k + m) + j] = a[k + j];
        a[(k + m) + j] -= t;
        a[k + j] += t;
        w *= w_m;
      }
      k += 2 * m;
    }
    m *= 2;
  }
}
pub fn fft(domain: GeneralEvaluationDomain<Fr>, a: &[G1Projective]) -> Vec<G1Projective>{
  let mut r = a.to_vec();
  fft_helper(&mut r, domain.group_gen(), domain.log_size_of_group());
  r
}
pub fn ifft(domain: GeneralEvaluationDomain<Fr>, a: &[G1Projective]) -> Vec<G1Projective>{
  let mut r = a.to_vec();
  fft_helper(&mut r, domain.group_gen_inv(), domain.log_size_of_group());
  r.iter_mut().for_each(|x| *x *= domain.size_inv());
  r
}
pub fn fft_in_place(domain: GeneralEvaluationDomain<Fr>, a: &mut [G1Projective]){
  fft_helper(a, domain.group_gen(), domain.log_size_of_group());
}
pub fn ifft_in_place(domain: GeneralEvaluationDomain<Fr>, a: &mut [G1Projective]){
  fft_helper(a, domain.group_gen_inv(), domain.log_size_of_group());
  a.iter_mut().for_each(|x| *x *= domain.size_inv());
}
