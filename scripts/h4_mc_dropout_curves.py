"""
H4 — MC-Dropout TP/FP Operating-Point Curves

The current Table 8 (tab:n5) caption claims MC-Dropout thresholds were
"calibrated to match the prototype-variance FP rate (≈21.7%)" but
Variant A reports FP=0.000, which is internally inconsistent.

This script sweeps η_var and η_ent over a fine grid and plots the full
FP-rate curve for each method, producing:
  (1) A JSON results file with (eta, fp_mean, fp_std, tp_mean, tp_std)
      for each method across the grid.
  (2) Printed summary showing each method's operating point at FP≈21.7%.

This makes the comparison defensible: if Variant A cannot reach 21.7% FP
at any threshold, we report that explicitly (the curve never crosses the
target FP line) rather than claiming a mis-calibrated threshold.

Results → /ephemeral/results/h4_mc_dropout_curves.json
         (used to update Table 8 caption and add a figure in the paper)
"""

import sys, os, json, random
sys.path.insert(0, "/home/ubuntu/Lung_cancer")
os.chdir("/home/ubuntu/Lung_cancer")

import torch
import numpy as np

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights
from src.utils.mc_dropout import (
    mc_forward_passes, prototype_variance_safety_check,
    mc_dropout_score_variance_check, mc_dropout_entropy_check,
)

DATA_DIR    = "/ephemeral/data/tbx11k"
CKPT_PATH   = "/ephemeral/checkpoints/tbx11k_no_C_novlm/csr_stage4_best.pt"
RESULTS_PATH= "/ephemeral/results/h4_mc_dropout_curves.json"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE  = 224
NUM_CONCEPTS= 3
NUM_CLASSES = 3
T_PASSES    = 30
DROPOUT_P   = 0.1
ETA_PROTO   = 0.05
N_TRIALS    = 5
MISC_RATE_P = 0.5
RANDOM_SEED = 42

os.makedirs("/ephemeral/results", exist_ok=True)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading model...")
model = CSRModel(
    num_concepts=NUM_CONCEPTS, num_prototypes=100, num_classes=NUM_CLASSES,
    proto_dim=256, use_gnn=True, use_uncertainty=True,
    use_vlm=False, pretrained_backbone=False,
).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()

# ── Build concept graph ────────────────────────────────────────────────────────
_train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train", image_size=IMAGE_SIZE)
_loader   = get_dataloader(_train_ds, 128, 4, shuffle=False)
_labels   = torch.cat([b["concept_labels"] for b in _loader]).to(DEVICE)
_ei, _ew  = build_cooccurrence_graph(_labels, threshold=0.10)
_ew_norm  = normalize_edge_weights(_ei, _ew, NUM_CONCEPTS)
model.set_concept_graph(_ei, _ew_norm)
del _train_ds, _loader, _labels

# ── Load data and pre-compute caches ──────────────────────────────────────────
bbox_ds     = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)
all_concepts= list(range(NUM_CONCEPTS))
samples     = []
for i in range(len(bbox_ds)):
    item = bbox_ds[i]
    for c_idx, bbox in {int(k): v for k, v in item["bbox"].items()}.items():
        samples.append((i, c_idx, bbox))
print(f"  {len(samples)} (image, concept, bbox) triples")

sim_cache, mc_cache = {}, {}
with torch.no_grad():
    for img_i in sorted(set(s[0] for s in samples)):
        item = bbox_ds[img_i]
        img  = item["image"].unsqueeze(0).to(DEVICE)
        f, _, _ = model.concept_model(img)
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)[0].cpu()
        sim_cache[img_i] = sm

print(f"Running MC-Dropout (T={T_PASSES}) passes...")
for img_i in sorted(sim_cache.keys()):
    item = bbox_ds[img_i]
    img  = item["image"].unsqueeze(0).to(DEVICE)
    mc_cache[img_i] = mc_forward_passes(model, img, T=T_PASSES, dropout_p=DROPOUT_P)
print("Pre-computation done.")


def eval_at_thresholds(eta_proto, eta_var, eta_ent, p, seed):
    rng = random.Random(seed)
    counts = {m: {"tp": 0, "fp": 0, "tp_total": 0, "fp_total": 0}
              for m in ["proto", "mc_var", "mc_ent"]}

    for img_i, concept_idx, bbox in samples:
        sm = sim_cache[img_i]
        mc = mc_cache[img_i]
        is_misdraw = rng.random() < p

        if is_misdraw:
            wrong   = [c for c in all_concepts if c != concept_idx]
            claimed = rng.choice(wrong)
            for m in counts: counts[m]["tp_total"] += 1
            if prototype_variance_safety_check(sm, bbox, concept_idx, claimed, IMAGE_SIZE, eta_proto):
                counts["proto"]["tp"] += 1
            if mc_dropout_score_variance_check(mc, bbox, concept_idx, claimed, IMAGE_SIZE, eta_var):
                counts["mc_var"]["tp"] += 1
            if mc_dropout_entropy_check(mc, eta_ent):
                counts["mc_ent"]["tp"] += 1
        else:
            claimed = concept_idx
            for m in counts: counts[m]["fp_total"] += 1
            if prototype_variance_safety_check(sm, bbox, concept_idx, claimed, IMAGE_SIZE, eta_proto):
                counts["proto"]["fp"] += 1
            if mc_dropout_score_variance_check(mc, bbox, concept_idx, claimed, IMAGE_SIZE, eta_var):
                counts["mc_var"]["fp"] += 1
            if mc_dropout_entropy_check(mc, eta_ent):
                counts["mc_ent"]["fp"] += 1

    out = {}
    for m, c in counts.items():
        out[m] = {
            "tp": c["tp"] / c["tp_total"] if c["tp_total"] else 0,
            "fp": c["fp"] / c["fp_total"] if c["fp_total"] else 0,
        }
    return out


# ── Threshold grids ────────────────────────────────────────────────────────────
ETA_VAR_GRID = [1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
ETA_ENT_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]

all_results = {"proto": [], "mc_var": [], "mc_ent": []}

# ── Prototype variance: sweep ETA_PROTO ────────────────────────────────────────
print("\nSweeping prototype variance threshold...")
for eta in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
    tp_list, fp_list = [], []
    for trial in range(N_TRIALS):
        r = eval_at_thresholds(eta, 0.01, 0.5, MISC_RATE_P, RANDOM_SEED + trial)
        tp_list.append(r["proto"]["tp"])
        fp_r = eval_at_thresholds(eta, 0.01, 0.5, 0.0, RANDOM_SEED + trial)
        fp_list.append(fp_r["proto"]["fp"])
    all_results["proto"].append({
        "eta": eta,
        "tp_mean": float(np.mean(tp_list)), "tp_std": float(np.std(tp_list)),
        "fp_mean": float(np.mean(fp_list)), "fp_std": float(np.std(fp_list)),
    })

# ── MC-Dropout score variance sweep ───────────────────────────────────────────
print("Sweeping MC-Dropout score variance threshold...")
for eta_var in ETA_VAR_GRID:
    tp_list, fp_list = [], []
    for trial in range(N_TRIALS):
        r = eval_at_thresholds(ETA_PROTO, eta_var, 0.5, MISC_RATE_P, RANDOM_SEED + trial)
        tp_list.append(r["mc_var"]["tp"])
        fp_r = eval_at_thresholds(ETA_PROTO, eta_var, 0.5, 0.0, RANDOM_SEED + trial)
        fp_list.append(fp_r["mc_var"]["fp"])
    all_results["mc_var"].append({
        "eta": eta_var,
        "tp_mean": float(np.mean(tp_list)), "tp_std": float(np.std(tp_list)),
        "fp_mean": float(np.mean(fp_list)), "fp_std": float(np.std(fp_list)),
    })

# ── MC-Dropout entropy sweep ───────────────────────────────────────────────────
print("Sweeping MC-Dropout entropy threshold...")
for eta_ent in ETA_ENT_GRID:
    tp_list, fp_list = [], []
    for trial in range(N_TRIALS):
        r = eval_at_thresholds(ETA_PROTO, 0.01, eta_ent, MISC_RATE_P, RANDOM_SEED + trial)
        tp_list.append(r["mc_ent"]["tp"])
        fp_r = eval_at_thresholds(ETA_PROTO, 0.01, eta_ent, 0.0, RANDOM_SEED + trial)
        fp_list.append(fp_r["mc_ent"]["fp"])
    all_results["mc_ent"].append({
        "eta": eta_ent,
        "tp_mean": float(np.mean(tp_list)), "tp_std": float(np.std(tp_list)),
        "fp_mean": float(np.mean(fp_list)), "fp_std": float(np.std(fp_list)),
    })

# ── Print operating points at FP≈21.7% ────────────────────────────────────────
TARGET_FP = 0.217
print(f"\n{'='*60}")
print(f"H4 Operating points closest to FP = {TARGET_FP:.3f}")
print(f"{'='*60}")

for method, rows in all_results.items():
    best = min(rows, key=lambda r: abs(r["fp_mean"] - TARGET_FP))
    print(f"  {method:>10}:  η={best['eta']:.5f}  "
          f"FP={best['fp_mean']:.3f}±{best['fp_std']:.3f}  "
          f"TP={best['tp_mean']:.3f}±{best['tp_std']:.3f}")

    # Also check if mc_var can reach target FP at all
    max_fp = max(r["fp_mean"] for r in rows)
    if method == "mc_var" and max_fp < TARGET_FP * 0.5:
        print(f"    ** mc_var max FP = {max_fp:.4f} — cannot reach target FP "
              f"(L2-norm suppresses score variance)")

with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nH4 done. Results → {RESULTS_PATH}")
print("Use these curves to update Table 8 caption in main.tex.")
