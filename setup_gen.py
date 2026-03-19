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

sessA = ZkTorchSession(yaml, fold=True) 
setups_bytes, models_enc_bytes = sessA.setup_and_export_models()              # compute once, cache, get bytes

# persist setups to a file / object store
with open("setups.affine.bin", "wb") as f:
    f.write(bytes(setups_bytes))
with open("models.enc.bin", "wb") as f:
    f.write(bytes(models_enc_bytes))