#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
use ark_ec::AffineRepr;
use ark_bls12_381::{Fr, G1Affine, G2Affine};
use ark_std::UniformRand;
use rand::{rngs::StdRng,SeedableRng};
use basic_block::*;
mod basic_block;
mod util;

fn test_basic_block<BB: BasicBlock>(model: &[Fr], inputs: &[&[Fr]]){
  let mut rng = StdRng::from_entropy();
  let x = Fr::rand(&mut rng);
  let mut xp = x;
  const N:usize = 1<<3;
  // Generate SRS
  let mut srs = ([G1Affine::generator() ; N], [G2Affine::generator() ; N+1]);
  for i in 1..N{
    srs.0[i]=(srs.0[i]*xp).into();
    xp*=x;
  }
  xp = x;
  for i in 1..N+1{
    srs.1[i]=(srs.1[i]*xp).into();
    xp*=x;
  }
  let srs: (&[G1Affine],&[G2Affine]) = (&(srs.0),&(srs.1));
  //Proof
  let output = BB::run(model,inputs);
  let mut model = Data::new(srs, model);
  let setup = BB::setup(srs,&mut model);
  let inputs : Vec<Data> = inputs.iter().map(|x| Data::new(srs,*x)).collect();
  let inputs : Vec<&Data> = inputs.iter().map(|x| x).collect();
  let output : Data = Data::new(srs,&output);
  let mut rng2 = rng.clone();
  let proof = BB::prove(srs,(&(setup.0),&(setup.1)),&model,&inputs,&output,&mut rng);
  let model = DataEnc::new(&model);
  let inputs : Vec<DataEnc> = inputs.iter().map(|x| DataEnc::new(*x)).collect();
  let inputs : Vec<&DataEnc> = inputs.iter().map(|x| x).collect();
  let output : DataEnc = DataEnc::new(&output);
  BB::verify(srs,&model,&inputs,&output,(&(proof.0),&(proof.1),&(proof.2)),&mut rng2);
}
fn main() {
  let mut rng = StdRng::from_entropy();
  const N:usize = 1<<3;
  const n:usize = 1<<2;
  let mut a = Vec::new();
  let mut b = Vec::new();
  for _ in 0..N{
    a.push(Fr::rand(&mut rng));
    b.push(Fr::rand(&mut rng));
  }
  let inputs:Vec<&[Fr]> = vec![&a,&b];
  test_basic_block::<AddBasicBlock>(&Vec::new(),&inputs);
  test_basic_block::<MulBasicBlock>(&Vec::new(),&inputs);
  b = a[..n].to_vec();
  let inputs:Vec<&[Fr]> = vec![&b];
  test_basic_block::<CQBasicBlock>(&a,&inputs);
}
