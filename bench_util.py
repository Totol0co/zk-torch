import time
from textwrap import dedent
from typing import Optional, Dict, Any

def bench_zktorch(
    onnx_model_path: str,
    input_json: Optional[str] = None,
    *,
    # These defaults match the small ptau file "challenge" present in the repo.
    # If you have a larger ptau, pass its path and logs accordingly.
    ptau_path: str = "challenge_0020",
    pow_len_log: int = 20,
    loaded_pow_len_log: int = 20,
    # Quantization / range defaults (safe, small demo settings)
    scale_factor_log: int = 4,
    cq_range_log: int = 8,
    cq_range_lower_log: int = 7,
    # Must match how the extension was built (features: fold / mock_prove etc.)
    fold: bool = True,
    # Enable on-disk caching for some CQ setups 
    enable_layer_setup: bool = False,
) -> Dict[str, Any]:
    """
    Benchmark ZKTorch setup/prove/verify times for a given ONNX and optional input JSON.

    Returns:
        {
            "t_setup_s": float,
            "t_prove_s": float,
            "t_verify_s": float,
            "verified": bool,
            "proofs_bytes_len": int,
            "acc_bytes_len": Optional[int],
            "out_plain_bytes_len": int
        }
    """
    # Build the minimal YAML config expected by ZkTorchSession.new/constructor
    # (same schema as config.yaml, but we inject your paths/knobs).
    # ZkTorchSession takes this YAML string directly.
    yaml_config = dedent(f"""
    task: python_api
    onnx:
      model_path: {onnx_model_path}
      input_path: {input_json if input_json is not None else ""}
    ptau:
      ptau_path: {ptau_path}
      pow_len_log: {pow_len_log}
      loaded_pow_len_log: {loaded_pow_len_log}
    sf:
      scale_factor_log: {scale_factor_log}
      cq_range_log: {cq_range_log}
      cq_range_lower_log: {cq_range_lower_log}
    prover:
      model_path: models
      setup_path: setups
      enc_model_path: modelsEnc
      enc_input_path: inputsEnc
      enc_output_path: outputsEnc
      proof_path: proofs
      acc_proof_path: acc_proofs
      final_proof_path: final_proofs
      enable_layer_setup: {"true" if enable_layer_setup else "false"}
    verifier:
      enc_model_path: modelsEnc
      enc_input_path: inputsEnc
      enc_output_path: outputsEnc
      proof_path: proofs
    """).strip()

    # Lazily import here so this file can be imported without pyzktorch being present.
    from pyzktorch import ZkTorchSession  # API: new(config_yaml, fold), setup(), prove(...), verify(...)
    # Create a fresh session (loads SRS from ptau, parses ONNX, pre-encodes model)
    # NOTE: 'fold' must match how you built the extension.
    # - setup() -> bytes
    # - prove(optional input_json) -> (proofs_bytes, acc_proofs_bytes_or_None, out_plain_bytes)
    # - verify(proofs_bytes, acc_bytes_or_None) -> bool
    # The session caches the encoded inputs/outputs from prove() for verify().
    # See test.py for reference usage in this repo.
    # (The API and return types are defined in src/python.rs)
    t0 = time.perf_counter()
    session = ZkTorchSession(yaml_config, fold=fold)
    t1 = time.perf_counter()

    # Setup
    t_setup_start = time.perf_counter()
    _setups_bytes = session.setup()
    t_setup_end = time.perf_counter()

    # Prove
    t_prove_start = time.perf_counter()
    proofs_bytes, acc_bytes_opt, out_plain = session.prove(input_json=input_json)
    t_prove_end = time.perf_counter()

    # Verify
    t_verify_start = time.perf_counter()
    verified = session.verify(proofs_bytes, acc_bytes_opt)
    t_verify_end = time.perf_counter()

    results = {
        "t_session_init_s": t1 - t0,
        "t_setup_s": t_setup_end - t_setup_start,
        "t_prove_s": t_prove_end - t_prove_start,
        "t_verify_s": t_verify_end - t_verify_start,
        "verified": bool(verified),
        "proofs_bytes_len": len(bytes(proofs_bytes)),
        "acc_bytes_len": (len(bytes(acc_bytes_opt)) if acc_bytes_opt is not None else None),
        "out_plain_bytes_len": len(bytes(out_plain)),
    }
    return results


if __name__ == "__main__":
    # Example run:
    # - If you use the tiny sample shipped in the repo, adjust the onnx path accordingly.
    # - Default ptau settings assume the "challenge" file in this repository (pow_len_log=7).
    res = bench_zktorch(
        onnx_model_path="sample.onnx",    # or your_model.onnx
        input_json="sample.json",         # or None -> random input (derived from ONNX input shape)
        ptau_path="challenge_0020",
        pow_len_log=20,
        loaded_pow_len_log=14,
        fold=True,                        # must match how you built the extension
        enable_layer_setup=False
    )
    print("Benchmark results:", res)