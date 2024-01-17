use powersoftau::parameters::{CeremonyParams, CheckForCorrectness, UseCompression};
use powersoftau::batched_accumulator::BatchedAccumulator;
use pairing_ce::bn256::Bn256;
use ff_ce::PrimeField;
use memmap::MmapOptions;
use std::fs::OpenOptions;
use ark_bn254::{Fq, Fq2, G1Affine, G2Affine};
use ark_ff::biginteger::BigInt;

pub fn load_file(filename: &str, n: usize) -> (Vec<G1Affine>,Vec<G2Affine>){
  let parameters = CeremonyParams::<Bn256>::new(n, 2097152);
  let reader = OpenOptions::new().read(true).open(filename).unwrap();
  assert!(reader.metadata().unwrap().len() == (parameters.accumulator_size as u64));
  let readable_map = unsafe{
    MmapOptions::new().map(&reader).unwrap()
  };
  let acc = BatchedAccumulator::deserialize(&readable_map, CheckForCorrectness::No, UseCompression::No, &parameters).unwrap();
  let temp  = acc.tau_powers_g1.iter().map(|x| G1Affine::new(Fq::new(BigInt::new(x.get_x().into_repr().0)),
                                                             Fq::new(BigInt::new(x.get_y().into_repr().0)))).collect();
  let temp2 = acc.tau_powers_g2.iter().map(|x| G2Affine::new(Fq2{c0: Fq::new(BigInt::new(x.get_x().c0.into_repr().0)),
                                                                 c1: Fq::new(BigInt::new(x.get_x().c1.into_repr().0))},
                                                             Fq2{c0: Fq::new(BigInt::new(x.get_y().c0.into_repr().0)),
                                                                 c1: Fq::new(BigInt::new(x.get_y().c1.into_repr().0))})).collect();
  return (temp, temp2);
}
