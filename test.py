from pyzktorch import ZkTorchSession

# Config parameters
yaml = """
task: python_api
onnx:
  model_path: sample.onnx
  input_path: sample.json # input_path is useless in the api version, should be put as parameter of prove()
ptau:
  ptau_path: challenge_0014
  pow_len_log: 14
  loaded_pow_len_log: 14
sf:
  scale_factor_log: 3
  cq_range_log: 6
  cq_range_lower_log: 5
prover:
  model_path: models
  setup_path: setups
  enc_model_path: modelsEnc
  enc_input_path: inputsEnc
  enc_output_path: outputsEnc
  proof_path: proofs
  acc_proof_path: acc_proofs
  final_proof_path: final_proofs
  enable_layer_setup: false
verifier:
  enc_model_path: modelsEnc
  enc_input_path: inputsEnc
  enc_output_path: outputsEnc
  proof_path: proofs
"""

session = ZkTorchSession(yaml, fold=True)       # build with "maturin develop --release -F fold" to use fold=true
setups_bytes = session.setup()
proofs_bytes, acc_bytes = session.prove(input_json="sample.json")  # input_json=None: gen a random input based on the onnx input shape
ok = session.verify(proofs_bytes, acc_bytes)


ok_input = session.input_verify("sample.json")
print("real input check: ", ok_input)

fake = session.input_verify("sample_fake.json")
print("fake input check: " + fake )