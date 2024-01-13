use ark_bls12_381::{Fr, G1Affine, G2Affine};
use rand::Rng;
use super::{BasicBlock,Data,DataEnc};

pub struct AddBasicBlock;
impl BasicBlock for AddBasicBlock{
  fn run(_model: &Vec<Fr>,
         inputs: &Vec<Vec<Fr>>) ->
         Vec<Fr>{
    let mut r = Vec::new();
    for i in 0..inputs[0].len(){
      r.push(inputs[0][i]+inputs[1][i]);
    }
    return r;
  }
  fn setup(_srs: (&Vec<G1Affine>,&Vec<G2Affine>),
           _model: &Data) ->
          (Vec<G1Affine>,Vec<G2Affine>){
    return (Vec::new(), Vec::new());
  }
  fn prove<R: Rng>(_srs: (&Vec<G1Affine>,&Vec<G2Affine>),
                   _setup: (&Vec<G1Affine>,&Vec<G2Affine>),
                   _model: &Data,
                   _inputs: &Vec<Data>,
                   _output: &Data,
                   _rng: &mut R) ->
                  (Vec<G1Affine>,Vec<G2Affine>,Vec<Fr>){
    return (Vec::new(), Vec::new(), Vec::new());
  }
  fn verify<R: Rng>(_srs: (&Vec<G1Affine>,&Vec<G2Affine>),
                    _model: &DataEnc,
                    inputs: &Vec<DataEnc>,
                    output: &DataEnc,
                    _proof: (&Vec<G1Affine>,&Vec<G2Affine>,&Vec<Fr>),
                    _rng: &mut R){
    // Verify f(x)+g(x)=h(x)
    let lhs = inputs[0].g1+inputs[1].g1;
    let rhs = output.g1;
    assert!(lhs==rhs);
  }
}

