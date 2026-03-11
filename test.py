from pyzktorch import ZkTorchSession

# YAML text (same schema as CLI config)
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
  enable_layer_setup: true
verifier:
  enc_model_path: modelsEnc
  enc_input_path: inputsEnc
  enc_output_path: outputsEnc
  proof_path: proofs
"""

sess = ZkTorchSession(yaml, fold=True)       # requires you built with -F fold
proofs_bytes, acc_bytes = sess.prove("sample.json")  # et non pas None
ok = sess.verify_with_input_json("sample.json", proofs_bytes, acc_bytes)
print(ok)  # True si tout est cohérent

"""
setups_bytes = sess.setup()
proofs_bytes, acc_bytes = sess.prove(input_json=None)  # or path to your JSON input
ok = sess.verify(proofs_bytes, acc_bytes)

# proofs_bytes, acc_bytes = sess.prove(None)

def flip_tail(b: bytes, k: int = 1) -> bytes:
    bb = bytearray(b)
    i = max(0, len(bb) - k)   # last byte by default
    bb[i] ^= 0x01             # flip one bit in the tail
    return bytes(bb)

bad_proofs = flip_tail(proofs_bytes)      # flip last byte: safe (no length header here)
ok = sess.verify(bad_proofs, acc_bytes)
print("Verified (corrupted proofs tail)?", ok)  # expect False

# If you’re using fold, you can also corrupt the acc proofs tail:
if acc_bytes is not None:
    bad_acc = flip_tail(acc_bytes)
    ok = sess.verify(proofs_bytes, bad_acc)
    print("Verified (corrupted acc tail)?", ok)  # expect False
    """