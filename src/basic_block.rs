use ark_ec::VariableBaseMSM;
use ark_poly::univariate::DensePolynomial;
use ark_poly::{GeneralEvaluationDomain, EvaluationDomain};
use ark_bls12_381::{Fr, G1Affine, G1Projective, G2Affine};
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
impl Data{
  pub fn new(srs:(&[G1Affine],&[G2Affine]), raw:&[Fr]) -> Data{
    let N = (*raw).len();
    let domain  = GeneralEvaluationDomain::<Fr>::new(N).unwrap();
    let f = DensePolynomial{coeffs: domain.ifft(raw)};
    let fx : G1Affine = G1Projective::msm_unchecked(&srs.0[..N], &f.coeffs).into();
    return Data{raw: raw.to_vec(), poly: f, g1: fx};
  }
}
pub struct DataEnc{
  pub len : usize,
  pub g1: G1Affine
}
impl DataEnc{
  pub fn new(data:&Data) -> DataEnc{
    return DataEnc{len: data.raw.len(), g1: data.g1};
  }
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


