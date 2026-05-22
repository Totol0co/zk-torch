#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import yaml
import glob
import shutil
import hashlib
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# --- (optionnel) dialogue pour choisir un dossier ---
def pick_folder_interactive() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Choisir le dossier contenant les modèles ONNX")
        return folder or ""
    except Exception:
        return ""

@dataclass
class BenchResult:
    model_name: str
    model_path: str
    out_dir: str
    setup_s: float
    prove_s: float
    verify_s: float
    final_proof_bytes: int
    proofs_bytes: int
    acc_proofs_bytes: int
    total_bytes: int
    status: str
    error: Optional[str] = None

# --- helpers ---
def safe_size(path: Optional[str]) -> int:
    try:
        if path and os.path.exists(path):
            return os.path.getsize(path)
        return 0
    except Exception:
        return 0

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def dump_yaml(cfg: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def derive_out_dir(root_out: str, onnx_path: str) -> str:
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    # ajoute un hash court pour éviter collisions
    h = hashlib.sha1(onnx_path.encode()).hexdigest()[:8]
    return os.path.join(root_out, f"{stem}_{h}")

def guess_input_json(onnx_path: str) -> Optional[str]:
    """Si un JSON de même nom que le ONNX existe à côté, l’utiliser."""
    candidate = os.path.splitext(onnx_path)[0] + ".json"
    return candidate if os.path.exists(candidate) else None

def make_per_model_config(base_cfg_path: str, onnx_path: str, out_root: str) -> str:
    cfg = load_yaml(base_cfg_path)
    out_dir = derive_out_dir(out_root, onnx_path)
    os.makedirs(out_dir, exist_ok=True)

    # ONNX model & input
    cfg.setdefault("onnx", {})
    cfg["onnx"]["model_path"] = onnx_path  # chemin absolu conseillé
    inp_guess = guess_input_json(onnx_path)
    if inp_guess:
        cfg["onnx"]["input_path"] = inp_guess  # sinon garde la valeur de base si absente

    # Prover / Verifier outputs: placer tout dans le sous-dossier par modèle
    for section in ("prover", "verifier"):
        cfg.setdefault(section, {})
    p = cfg["prover"]
    v = cfg["verifier"]

    # sous-répertoires ou fichiers (le code Rust ouvre des fichiers, on laisse des chemins “fichiers”)
    p["model_path"]         = os.path.join(out_dir, "models")
    p["setup_path"]         = os.path.join(out_dir, "setups")
    p["enc_model_path"]     = os.path.join(out_dir, "modelsEnc")
    p["enc_input_path"]     = os.path.join(out_dir, "inputsEnc")
    p["enc_output_path"]    = os.path.join(out_dir, "outputsEnc")
    p["proof_path"]         = os.path.join(out_dir, "proofs")
    p["acc_proof_path"]     = os.path.join(out_dir, "acc_proofs")
    p["final_proof_path"]   = os.path.join(out_dir, "final_proofs")

    # si tu as ajouté l’export des sorties “plaintext” :
    p["final_output_field_json_path"] = os.path.join(out_dir, "final_outputs_field.json")
    p["final_output_float_json_path"] = os.path.join(out_dir, "final_outputs_float.json")
    p["final_output_bin_path"]        = os.path.join(out_dir, "final_outputs.bin")

    # côté vérif, pointer vers les mêmes encodings
    v["enc_model_path"]  = p["enc_model_path"]
    v["enc_input_path"]  = p["enc_input_path"]
    v["enc_output_path"] = p["enc_output_path"]
    v["proof_path"]      = p["proof_path"]

    # écrire la config temp
    cfg_path = os.path.join(out_dir, "config.yaml")
    dump_yaml(cfg, cfg_path)
    return cfg_path


def run_in_subprocess(config_path: str, prove_runs: int, verify_runs: int) -> Dict:
    inline = r"""
import sys, time, json, os, yaml, statistics
import pyzktorch as zkt
import time
cfg_path = sys.argv[1]
prove_runs = int(sys.argv[2])
verify_runs = int(sys.argv[3])

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
p = cfg.get("prover", {})

# --- SETUP (1 fois) ---
t0 = time.perf_counter()
zkt.setup(cfg_path)
t1 = time.perf_counter()
setup_s = t1 - t0

time.sleep(1)

# --- PROVE (N fois) ---
prove_times = []
for i in range(prove_runs):
    try:
        t0 = time.perf_counter()
        zkt.prove(cfg_path)
        t1 = time.perf_counter()
        prove_times.append(t1 - t0)
        time.sleep(0.5)
    except Exception:
        pass  # ignore run en erreur

# --- VERIFY (M fois) ---
verify_times = []
for i in range(verify_runs):
    try:
        t0 = time.perf_counter()
        zkt.verify(cfg_path)
        t1 = time.perf_counter()
        verify_times.append(t1 - t0)
        time.sleep(0.5)
    except Exception:
        pass

sizes = {
  "final_proof_path": os.path.getsize(p["final_proof_path"]) if os.path.exists(p["final_proof_path"]) else 0,
  "proof_path":       os.path.getsize(p["proof_path"])       if os.path.exists(p["proof_path"]) else 0,
  "acc_proof_path":   os.path.getsize(p["acc_proof_path"])   if os.path.exists(p["acc_proof_path"]) else 0,
}

res = {
  "setup_s": setup_s,
  "prove_times": prove_times,
  "verify_times": verify_times,
  "sizes": sizes,
}
print("__RESULT__:" + json.dumps(res))
"""
    proc = subprocess.run(
        [sys.executable, "-c", inline, config_path, str(prove_runs), str(verify_runs)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__:"):
            return json.loads(line[len("__RESULT__:"):])

    raise RuntimeError("No result JSON found")


def bench_folder(base_cfg_path: str, models_dir: str, out_root: str, pruns: int, vruns: int) -> List[BenchResult]:
    onnx_files = sorted(glob.glob(os.path.join(models_dir, "*.onnx")))
    if not onnx_files:
        print(f"Aucun .onnx trouvé dans: {models_dir}")
        return []

    results: List[BenchResult] = []
    os.makedirs(out_root, exist_ok=True)
    
    total = len(onnx_files)
    for idx, onnx_path in enumerate(onnx_files, start=1):
        model_name = os.path.basename(onnx_path)
        print(f"\n[{idx}/{total}] Traitement de : {model_name}")

        model_name = os.path.basename(onnx_path)
        out_dir = derive_out_dir(out_root, onnx_path)
        # repart sur un dossier propre si déjà présent
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)

        try:
            cfg_path = make_per_model_config(base_cfg_path, onnx_path, out_root)
            # lance les phases en sous-processus
            stats = run_in_subprocess(cfg_path, pruns, vruns)

            # tailles
            sizes = stats.get("sizes", {})
            final_bytes = int(sizes.get("final_proof_path", 0))
            proof_bytes = int(sizes.get("proof_path", 0))
            acc_bytes   = int(sizes.get("acc_proof_path", 0))

            # définition de la "taille de la preuve"
            # - si final_proof existe (fold), on le privilégie
            # - sinon on somme proof + acc_proofs
            proof_total = final_bytes if final_bytes > 0 else (proof_bytes + acc_bytes)
            
            prove_times = stats.get("prove_times", [])
            verify_times = stats.get("verify_times", [])

            if not prove_times or not verify_times:
                raise RuntimeError("No valid prove/verify runs")

            prove_avg = sum(prove_times) / len(prove_times)
            verify_avg = sum(verify_times) / len(verify_times)

            prove_med = sorted(prove_times)[len(prove_times)//2]
            verify_med = sorted(verify_times)[len(verify_times)//2]

            
            results.append(BenchResult(
                model_name=model_name,
                model_path=os.path.abspath(onnx_path),
                out_dir=out_dir,
                setup_s=float(stats["setup_s"]),
                prove_s=prove_med,      # ✅ médiane (plus stable)
                verify_s=verify_med,
                final_proof_bytes=final_bytes,
                proofs_bytes=proof_bytes,
                acc_proofs_bytes=acc_bytes,
                total_bytes=proof_total,
                status="ok",
            ))

            
        except Exception as e:
            results.append(BenchResult(
                model_name=model_name,
                model_path=os.path.abspath(onnx_path),
                out_dir=out_dir,
                setup_s=0.0, prove_s=0.0, verify_s=0.0,
                final_proof_bytes=0, proofs_bytes=0, acc_proofs_bytes=0, total_bytes=0,
                status="error",
                error=str(e),
            ))
        print(results)
    return results

def print_summary(results: List[BenchResult]):
    # affichage lisible
    header = f"{'Model':35s} | {'Setup (s)':>9} | {'Prove (s)':>9} | {'Verify (s)':>10} | {'Proof size (bytes)':>18} | Status"
    print(header)
    print("-"*len(header))
    for r in results:
        print(f"{r.model_name:35s} | {r.setup_s:9.3f} | {r.prove_s:9.3f} | {r.verify_s:10.3f} | {r.total_bytes:18d} | {r.status}")

def save_json(results: List[BenchResult], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Bench ZKTorch sur un dossier de modèles ONNX")
    ap.add_argument("--config", "-c", required=True, help="Chemin vers le config.yaml de base")
    ap.add_argument("--dir", "-d", help="Dossier contenant les .onnx (si absent, dialogue)")
    ap.add_argument("--out", "-o", default="bench_out", help="Dossier racine pour sorties/rapports")
    ap.add_argument("--pruns", type=int, default=1, help="Nombre de runs de prove")
    ap.add_argument("--vruns", type=int, default=1, help="Nombre de runs de verify")

    args = ap.parse_args()

    models_dir = args.dir or pick_folder_interactive()
    if not models_dir:
        print("Aucun dossier sélectionné.")
        sys.exit(1)
    if not os.path.isdir(models_dir):
        print(f"Dossier invalide: {models_dir}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    results = bench_folder(args.config, models_dir, args.out, args.pruns, args.vruns)
    print_summary(results)

    report_path = os.path.join(args.out, "bench_report.json")
    save_json(results, report_path)
    print(f"\nRapport JSON écrit dans: {report_path}")

if __name__ == "__main__":
    main()