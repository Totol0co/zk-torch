use ark_bls12_381::{Fr, G1Affine, G2Affine};
use rand::Rng;
use super::{BasicBlock,Data,DataEnc};

pub struct AddBasicBlock;
impl BasicBlock for AddBasicBlock{
  fn run(_: &[Fr],
         inputs: &[&[Fr]]) ->
         Vec<Fr>{
    let mut r = Vec::new();
    for i in 0..inputs[0].len(){
      r.push(inputs[0][i]+inputs[1][i]);
    }
    return r;
  }
  fn setup(_: (&[G1Affine],&[G2Affine]),
           _: &mut Data) ->
          (Vec<G1Affine>,Vec<G2Affine>){
    return (Vec::new(), Vec::new());
  }
  fn prove<R: Rng>(_: (&[G1Affine],&[G2Affine]),
                   _: (&[G1Affine],&[G2Affine]),
                   _: &Data,
                   _: &[&Data],
                   _: &Data,
                   _: &mut R) ->
                  (Vec<G1Affine>,Vec<G2Affine>){
    return (Vec::new(), Vec::new());
  }
  fn verify<R: Rng>(_: (&[G1Affine],&[G2Affine]),
                    _: &DataEnc,
                    inputs: &[&DataEnc],
                    output: &DataEnc,
                    _: (&[G1Affine],&[G2Affine]),
                    _: &mut R){
    // Verify f(x)+g(x)=h(x)
    let lhs = inputs[0].g1+inputs[1].g1;
    let rhs = output.g1;
    assert!(lhs==rhs);
  }
}

