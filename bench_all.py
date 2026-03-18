import os, sys, json, subprocess
from typing import Optional

def find_input_json(onnx_path: str) -> Optional[str]:
    j = os.path.splitext(onnx_path)[0] + ".json"
    return j if os.path.exists(j) else None


def _extract_last_json_line(s: str) -> Optional[str]:
    # Scan from last line upwards; return the first line that looks like JSON.
    for line in reversed(s.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return line
    return None


def run_one(onnx_path, input_json, fold):
    cmd = [
        sys.executable, "bench_one.py",
        "--onnx", onnx_path,
        "--pow-len-log", "20",
        "--loaded-pow-len-log", "13",
        "--scale-factor-log", "4",
        "--cq-range-log", "12",
        "--cq-range-lower-log", "11",
        "--ptau", "challenge_0020",
    ]
    
    if fold:
        cmd += ["--fold"]
    # No input_json passed because you rely on ZKTorch random inputs.

    out = subprocess.run(cmd, capture_output=True, text=True)

    # Debug (optional): uncomment if you need to inspect failures.
    # print("----- STDOUT -----\n", out.stdout)
    # print("----- STDERR -----\n", out.stderr)

    last_json = _extract_last_json_line(out.stdout)
    if out.returncode != 0 or not last_json:
        # Return a minimal error record keyed by model name
        return {
            "model": os.path.basename(onnx_path),
            "error": f"exit={out.returncode}",
        }

    try:
        return json.loads(last_json)
    except json.JSONDecodeError:
        return {
            "model": os.path.basename(onnx_path),
            "error": "JSONDecodeError",
        }


def benchmark_folder(folder: str, fold: bool = True) -> dict:
    results = {}
    onnx_files = sorted([f for f in os.listdir(folder) if f.endswith(".onnx")])
    if not onnx_files:
        print("No .onnx files in", folder)
        return results

    for fname in onnx_files:
        onnx_path = os.path.join(folder, fname)
        input_json = find_input_json(onnx_path)
        print(f"\n=== Benchmarking {fname} ===")
        res = run_one(onnx_path, input_json, fold)
        results[fname] = res
        print(" -> Done:", res)
    return results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True)
    p.add_argument("--out", default="bench_results.json")
    p.add_argument("--fold", action="store_true")
    args = p.parse_args()

    results = benchmark_folder(args.folder, fold=args.fold)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to", args.out)
