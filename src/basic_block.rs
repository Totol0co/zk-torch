use ark_poly::univariate::DensePolynomial;
use ark_bls12_381::{Fr, G1Affine, G2Affine};
use rand::Rng;
pub use cq::CQBasicBlock;
pub use mul::MulBasicBlock;
pub use add::AddBasicBlock;
pub mod cq;
pub mod mul;
pub mod add;

pub struct Data{
  pub raw : Vec<Fr>,
  pub poly: DensePolynomial<Fr>,
  pub g1: G1Affine
}
pub struct DataEnc{
  pub len : usize,
  pub g1: G1Affine
}
pub trait BasicBlock{
  fn run(model: &[Fr],
         inputs: &[&[Fr]]) ->
         Vec<Fr>;
  fn setup(srs: (&[G1Affine],&[G2Affine]),
           model: &mut Data) ->
          (Vec<G1Affine>,Vec<G2Affine>);
  fn prove<R: Rng>(srs: (&[G1Affine],&[G2Affine]),
                   setup: (&[G1Affine],&[G2Affine]),
                   model: &Data,
                   inputs: &[&Data],
                   output: &Data,
                   rng: &mut R) ->
                  (Vec<G1Affine>,Vec<G2Affine>,Vec<Fr>);
  fn verify<R: Rng>(srs: (&[G1Affine],&[G2Affine]),
                    model: &DataEnc,
                    inputs: &[&DataEnc],
                    output: &DataEnc,
                    proof: (&[G1Affine],&[G2Affine],&[Fr]),
                    rng: &mut R);
}


