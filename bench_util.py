import time
from textwrap import dedent
from typing import Optional, Dict, Any
import numpy as np
import struct
from pyzktorch import ZkTorchSession
import onnx
from onnx import TensorProto


# ---------- ONNX dtype mapping ----------

try:
    BF16_DTYPE = np.dtype("bfloat16")  # NumPy >= 1.20/2.0
except TypeError:
    BF16_DTYPE = None  # Fallback: cast BF16 to float32.

ONNX_TO_NUMPY_DTYPE = {
    TensorProto.FLOAT:    np.float32,
    TensorProto.DOUBLE:   np.float64,
    TensorProto.FLOAT16:  np.float16,
    TensorProto.BFLOAT16: BF16_DTYPE if BF16_DTYPE is not None else np.float32,
    TensorProto.BOOL:     np.bool_,
    TensorProto.INT8:     np.int8,
    TensorProto.INT16:    np.int16,
    TensorProto.INT32:    np.int32,
    TensorProto.INT64:    np.int64,
    TensorProto.UINT8:    np.uint8,
    TensorProto.UINT16:   np.uint16,
    TensorProto.UINT32:   np.uint32,
    TensorProto.UINT64:   np.uint64,
}

def _elemtype_to_np_dtype(elem_type: int) -> np.dtype:
    if elem_type not in ONNX_TO_NUMPY_DTYPE:
        raise ValueError(f"Unsupported ONNX elem_type: {elem_type}")
    return ONNX_TO_NUMPY_DTYPE[elem_type]

def _onnx_out_elem_types(onnx_model_path: str):
    m = onnx.load(onnx_model_path)
    outs = list(m.graph.output)
    return [vi.type.tensor_type.elem_type for vi in outs]

def decode_out_plain(session, out_plain: bytes, onnx_model_path: str):
    """
    Decode ZkTorchSession.prove(...)'s 'out_plain' into properly typed NumPy arrays,
    using the ONNX model's output dtypes and shapes.

    Returns: List[np.ndarray] in the same order as model.graph.output.
    """
    model = onnx.load(onnx_model_path)
    onnx_outputs = list(model.graph.output)  # preserves order

    # Raw decode from Rust: List[(shape: List[int], flat_ints: List[int])]
    decoded = session.decode_outputs_plain_to_raw(out_plain)

    if len(decoded) != len(onnx_outputs):
        raise ValueError(
            f"Mismatch between decoded outputs ({len(decoded)}) and ONNX outputs ({len(onnx_outputs)}). "
            f"Ensure Graph.outputs is filled and 'out_plain' corresponds to final graph outputs."
        )

    sf_log = session.get_scale_factor_log()
    sf = float(1 << sf_log)

    typed = []
    for (shape, flat_ints), vi in zip(decoded, onnx_outputs):
        elem_type = vi.type.tensor_type.elem_type
        np_dtype = _elemtype_to_np_dtype(elem_type)

        # Start from int64 then cast appropriately
        arr = np.asarray(flat_ints, dtype=np.int64).reshape(tuple(shape))

        if np.issubdtype(np_dtype, np.floating):
            # Dequantize floats
            arr = (arr.astype(np.float32) / sf).astype(np_dtype)
        elif np_dtype == np.bool_:
            arr = (arr != 0)
        else:
            # Integers -> direct cast
            arr = arr.astype(np_dtype)

        typed.append(arr)

    return typed

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
    

    typed_outputs = decode_out_plain(session, out_plain, onnx_model_path=onnx_model_path)
    for i, arr in enumerate(typed_outputs):
        print(f"[final output #{i}] dtype={arr.dtype}, shape={arr.shape}\n{arr}\n")

    # Real output check (should be True)
    ok_output = session.output_verify(out_plain)
    print("real output check:", ok_output)

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