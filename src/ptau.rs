use ark_bn254::{Fq, Fq2, G1Affine, G2Affine};
use ark_ff::PrimeField;
use std::fs;
use rayon::prelude::*;

pub fn load_file(filename: &str, n: usize) -> (Vec<G1Affine>,Vec<G2Affine>){
  let bytes = fs::read(filename).unwrap();

  let powers_length=1<<n;
  let powers_g1_length=(powers_length<<1)-1;

  let g1 = (0..powers_g1_length).into_par_iter().map(|i|{
    let start = 64 + i*64;
    let x = Fq::from_be_bytes_mod_order(&bytes[start..start+32]);
    let y = Fq::from_be_bytes_mod_order(&bytes[start+32..start+64]);
    G1Affine::new_unchecked(x,y)
  }).collect();

  let g2 = (0..powers_length).into_par_iter().map(|i|{
    let start = 64 + 64*powers_g1_length + 128*i;
    let a = Fq::from_be_bytes_mod_order(&bytes[start..start+32]);
    let b = Fq::from_be_bytes_mod_order(&bytes[start+32..start+64]);
    let c = Fq::from_be_bytes_mod_order(&bytes[start+64..start+96]);
    let d = Fq::from_be_bytes_mod_order(&bytes[start+96..start+128]);
    G2Affine::new_unchecked(Fq2{c0:b, c1:a}, Fq2{c0:d, c1:c})
  }).collect();

  return (g1,g2);
}
