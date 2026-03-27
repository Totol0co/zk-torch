import argparse, json, os
from bench_util import bench_zktorch  

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--input", default=None)
    p.add_argument("--ptau", default="challenge")
    p.add_argument("--pow-len-log", type=int, default=7)
    p.add_argument("--loaded-pow-len-log", type=int, default=7)
    p.add_argument("--scale-factor-log", type=int, default=4)
    p.add_argument("--cq-range-log", type=int, default=6)
    p.add_argument("--cq-range-lower-log", type=int, default=5)
    p.add_argument("--fold", action="store_true")
    p.add_argument("--enable-layer-setup", action="store_true")
    args = p.parse_args()

    r = bench_zktorch(
        onnx_model_path=args.onnx,
        input_json=args.input,
        ptau_path=args.ptau,
        pow_len_log=args.pow_len_log,
        loaded_pow_len_log=args.loaded_pow_len_log,
        scale_factor_log=args.scale_factor_log,
        cq_range_log=args.cq_range_log,
        cq_range_lower_log=args.cq_range_lower_log,
        fold=args.fold,
        enable_layer_setup=args.enable_layer_setup,
    )
    print(r)

    # Emit only the fields you want, in ONE final JSON line.
    payload = {
        "model": os.path.basename(args.onnx),
        "t_setup_s": r.get("t_setup_s"),
        "t_prove_s": r.get("t_prove_s"),
        "t_verify_s": r.get("t_verify_s"),
    }
    print(json.dumps(payload), flush=True)

if __name__ == "__main__":
    main()