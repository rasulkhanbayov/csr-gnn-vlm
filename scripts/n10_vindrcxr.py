"""
N10 — VinDr-CXR Cross-Dataset Generalisation Experiment

Trains four GRAPE variants on VinDr-CXR (14-concept, 18k images) to test
whether the improvements that work on TBX11K generalise to a different
multi-finding CXR dataset with bbox annotations:

  (a) GRAPE full         (GNN + Uncertainty + VLM)
  (b) GRAPE w/o VLM      (GNN + Uncertainty only)
  (c) GRAPE w/o GNN      (Uncertainty + VLM only)
  (d) CSR baseline       (none of A/B/C)

Protocol: 5-stage curriculum, identical hyper-parameters to TBX11K run
except graph_threshold=0.05 (VinDr findings co-occur more densely than TB).
Reports: Macro-F1 (binary no_finding / finding), Pointing Game accuracy on
bbox_eval split (up to 500 annotated images).

Prerequisites: VinDr-CXR must be downloaded first via:
  python scripts/download_vindrcxr.py

Results → Table N10 / §4.7 in paper (cross-dataset generalisation).
"""

import sys, os, json, subprocess
sys.path.insert(0, "/home/ubuntu/Lung_cancer")
os.chdir("/home/ubuntu/Lung_cancer")

import torch
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "/ephemeral/data/vindrcxr"
CKPT_BASE    = "/ephemeral/checkpoints/vindrcxr"
RESULTS_PATH = "/ephemeral/results/n10_vindrcxr.json"
LOG_DIR      = "/ephemeral/logs"
DEVICE_STR   = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG       = "configs/vindrcxr_config.yaml"

os.makedirs(CKPT_BASE, exist_ok=True)
os.makedirs("/ephemeral/results", exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ── Verify dataset is downloaded ──────────────────────────────────────────────
if not os.path.exists(os.path.join(DATA_DIR, "labels.csv")):
    print("ERROR: VinDr-CXR not found at", DATA_DIR)
    print("Run: python scripts/download_vindrcxr.py")
    sys.exit(1)


# ── Ablation variants ─────────────────────────────────────────────────────────
VARIANTS = [
    {
        "name":       "grape_full",
        "label":      "GRAPE (GNN+Unc+VLM)",
        "extra_args": [],
        "ckpt_dir":   f"{CKPT_BASE}/grape_full",
    },
    {
        "name":       "grape_no_vlm",
        "label":      "GRAPE w/o VLM",
        "extra_args": ["--no_vlm"],
        "ckpt_dir":   f"{CKPT_BASE}/grape_no_vlm",
    },
    {
        "name":       "grape_no_gnn",
        "label":      "GRAPE w/o GNN",
        "extra_args": ["--no_gnn"],
        "ckpt_dir":   f"{CKPT_BASE}/grape_no_gnn",
    },
    {
        "name":       "csr_baseline",
        "label":      "CSR baseline",
        "extra_args": ["--no_gnn", "--no_uncertainty", "--no_vlm"],
        "ckpt_dir":   f"{CKPT_BASE}/csr_baseline",
    },
]


def run_variant(variant: dict) -> dict:
    """Launch train.py for this variant and parse final results."""
    os.makedirs(variant["ckpt_dir"], exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"vindrcxr_{variant['name']}.log")

    cmd = [
        sys.executable, "train.py",
        "--dataset",        "vindrcxr",
        "--data_dir",       DATA_DIR,
        "--config",         CONFIG,
        "--checkpoint_dir", variant["ckpt_dir"],
        "--num_workers",    "8",
        "--batch_size",     "128",
        "--seed",           "42",
    ] + variant["extra_args"]

    print(f"\n{'='*60}")
    print(f"  Running: {variant['label']}")
    print(f"  Log: {log_path}")
    print(f"{'='*60}")

    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd="/home/ubuntu/Lung_cancer",
        )

    # Parse result from log
    results = {"macro_f1": None, "pg": None, "returncode": proc.returncode}
    try:
        with open(log_path) as f:
            lines = f.readlines()

        for line in reversed(lines):
            line = line.strip()
            if line.startswith("macro_f1:"):
                results["macro_f1"] = float(line.split(":")[1].strip())
            elif line.startswith("pointing_game:"):
                results["pg"] = float(line.split(":")[1].strip())
            if results["macro_f1"] is not None and results["pg"] is not None:
                break
    except Exception as e:
        print(f"  Warning: could not parse results — {e}")

    print(f"  → F1={results['macro_f1']}  PG={results['pg']}  (rc={proc.returncode})")
    return results


# ── Run all variants ──────────────────────────────────────────────────────────
all_results = {}
for variant in VARIANTS:
    all_results[variant["name"]] = run_variant(variant)
    all_results[variant["name"]]["label"] = variant["label"]

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("N10 Results: VinDr-CXR Ablation")
print(f"{'='*65}")
print(f"{'Variant':>30}  {'Macro F1':>10}  {'PG':>8}")
print("-" * 55)
for name, r in all_results.items():
    f1_str = f"{r['macro_f1']:.4f}" if r["macro_f1"] is not None else "   N/A"
    pg_str = f"{r['pg']:.4f}" if r["pg"] is not None else "   N/A"
    print(f"{r['label']:>30}  {f1_str:>10}  {pg_str:>8}")

with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nN10 done. Results → {RESULTS_PATH}")
