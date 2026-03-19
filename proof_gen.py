from pyzktorch import ZkTorchSession

yaml = """
task: python_api
onnx:
    model_path: sample.onnx
    input_path: sample.json
ptau:
    ptau_path: challenge_0014
    pow_len_log: 14
    loaded_pow_len_log: 14
sf:
    scale_factor_log: 4
    cq_range_log: 7
    cq_range_lower_log: 6
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

sessB = ZkTorchSession(yaml, fold=True)

# load affine setups shipped from Machine A
with open("setups.affine.bin", "rb") as f:
    setups_bytes = f.read()

# load models enc
with open("models.enc.bin", "rb") as f:
    models_enc_bytes = f.read()

sessB.load_setups_and_models_from_bytes(setups_bytes, models_enc_bytes)

# now prove WITHOUT recomputing setup
proofs_bytes, acc_bytes, out_plain = sessB.prove(input_json="sample.json")
ok = sessB.verify(proofs_bytes, acc_bytes)