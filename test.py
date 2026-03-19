# test_end_to_end.py
from pyzktorch import ZkTorchSession
import numpy as np
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

# ---------- Decoding: out_plain (bytes) -> typed NumPy arrays ----------

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

# ---------- Encoding: typed outputs -> out_plain (bytes) ----------

def typed_to_out_plain(session, typed_outputs, onnx_model_path: str) -> bytes:
    """
    Convert typed NumPy tensors (same order as model.graph.output) into a
    bincode 'out_plain' compatible with output_verify_from_plain_bytes(...).
    """
    elem_types = _onnx_out_elem_types(onnx_model_path)
    if len(elem_types) != len(typed_outputs):
        raise ValueError(
            f"typed_outputs ({len(typed_outputs)}) must match ONNX outputs ({len(elem_types)}). "
            f"Order must be the same as model.graph.output."
        )

    sf_log = session.get_scale_factor_log()
    sf = float(1 << sf_log)

    shapes = []
    flat_ints = []  # List[List[int64]]

    for arr, elem_type in zip(typed_outputs, elem_types):
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        shapes.append(list(arr.shape))

        if elem_type in (TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.FLOAT16, TensorProto.BFLOAT16):
            # Quantize floats: value * 2^sf_log -> round -> int64
            q = np.rint(arr.astype(np.float32) * sf).astype(np.int64)
            flat_ints.append(q.reshape(-1).tolist())
        elif elem_type == TensorProto.BOOL:
            q = (arr.astype(np.bool_) != 0).astype(np.int64)
            flat_ints.append(q.reshape(-1).tolist())
        else:
            # Integer outputs: cast to int64 directly
            q = arr.astype(np.int64, copy=False)
            flat_ints.append(q.reshape(-1).tolist())

    # Call Rust to serialize (Vec<PlainNdArray>) -> bincode bytes
    pybytes = session.encode_outputs_plain_from_raw(shapes, flat_ints)
    return bytes(pybytes)

# ---------- Test: end-to-end ----------

if __name__ == "__main__":
    # Config parameters
    yaml = """
    task: python_api
    onnx:
      model_path: onnx/unit/test/sign.onnx
      input_path: sample.json
    ptau:
      ptau_path: challenge_0014
      pow_len_log: 14
      loaded_pow_len_log: 14
    sf:
      scale_factor_log: 4
      cq_range_log: 13
      cq_range_lower_log: 12
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

    onnx_model_path = "onnx/unit/test/sign.onnx"

    # Create session, setup, prove, verify
    session = ZkTorchSession(yaml, fold=True)
    _ = session.setup()
    proofs_bytes, acc_bytes, out_plain = session.prove(input_json=None)
    ok = session.verify(proofs_bytes, acc_bytes)
    print("verify(proofs, acc):", ok)

    # Decode the final public outputs into typed NumPy arrays
    typed_outputs = decode_out_plain(session, out_plain, onnx_model_path=onnx_model_path)
    for i, arr in enumerate(typed_outputs):
        print(f"[final output #{i}] dtype={arr.dtype}, shape={arr.shape}\n{arr}\n")

    # Real output check (should be True)
    ok_output = session.output_verify(out_plain)
    print("real output check:", ok_output)

    # Build fake output
    typed_fake = []
    for arr in typed_outputs:
        arr_fake = arr.copy()
        if arr_fake.size > 0:
            if np.issubdtype(arr_fake.dtype, np.floating):
                arr_fake.flat[0] += 1.0
            elif arr_fake.dtype == np.bool_:
                arr_fake.flat[0] = ~arr_fake.flat[0]
            else:
                # integer types
                arr_fake.flat[0] = arr_fake.flat[0] + 1
        typed_fake.append(arr_fake)

    # Encode fake -> binary and check (should be False)
    out_plain_fake = typed_to_out_plain(session, typed_fake, onnx_model_path=onnx_model_path)
    ok_fake = session.output_verify(out_plain_fake)
    print("fake output check:", ok_fake)

    # round‑trip: re-encode true typed output and check True (should be True)
    out_plain_true = typed_to_out_plain(session, typed_outputs, onnx_model_path=onnx_model_path)
    ok_true_roundtrip = session.output_verify(out_plain_true)
    print("roundtrip true output check:", ok_true_roundtrip)

    # Input checks
    ok_input = session.input_verify("sample.json")
    fake_input = session.input_verify("sample_fake.json")
    print("real input check:", ok_input)
    print("fake input check:", fake_input)