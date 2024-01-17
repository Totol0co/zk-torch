#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use ark_bn254::{Fr, G1Affine, G2Affine};
use ark_std::UniformRand;
use rand::{rngs::StdRng,SeedableRng};
use basic_block::*;
mod basic_block;
mod util;
mod ptau;

fn test_basic_block<BB: BasicBlock>(srs: (&Vec<G1Affine>,&Vec<G2Affine>), model: &Vec<Fr>, inputs: &Vec<Vec<Fr>>){
  let mut rng = StdRng::from_entropy();
  let output = BB::run(model,inputs);
  let model = Data::new(srs, model);
  let setup = BB::setup(srs,&model);
  let inputs = inputs.iter().map(|x| Data::new(srs,x)).collect();
  let output = Data::new(srs,&output);
  let mut rng2 = rng.clone();
  let proof = BB::prove(srs,(&(setup.0),&(setup.1)),&model,&inputs,&output,&mut rng);
  let model = DataEnc::new(&model);
  let inputs = inputs.iter().map(|x| DataEnc::new(x)).collect();
  let output = DataEnc::new(&output);
  BB::verify(srs,&model,&inputs,&output,(&(proof.0),&(proof.1),&(proof.2)),&mut rng2);
}
fn main() {
  let srs = ptau::load_file("challenge",7);
  let srs = (&srs.0,&srs.1);
  const N:usize = 1<<6;
  const n:usize = 1<<3;
  let mut rng = StdRng::from_entropy();
  let mut a = Vec::new();
  let mut b = Vec::new();
  for _ in 0..N{
    a.push(Fr::rand(&mut rng));
    b.push(Fr::rand(&mut rng));
  }
  test_basic_block::<AddBasicBlock>(srs,&Vec::new(),&vec![a.clone(),b.clone()]);
  test_basic_block::<MulBasicBlock>(srs,&Vec::new(),&vec![a.clone(),b.clone()]);
  test_basic_block::<CQBasicBlock>(srs,&a,&vec![a[..n].to_vec()]);
  test_basic_block::<CQLinBasicBlock>(srs,&a,&vec![b[..n].to_vec()]);
}
