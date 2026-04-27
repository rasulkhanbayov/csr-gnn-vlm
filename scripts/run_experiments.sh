#!/bin/bash
# Full experiment pipeline: download data → train models → run C7/C8/N3 experiments
# Usage: bash scripts/run_experiments.sh 2>&1 | tee /ephemeral/logs/experiments.log

set -e
CONDA_LIB=/opt/anaconda3/2024.02-1/conda_envs/ml_dl_gpu_base/lib
export LD_LIBRARY_PATH=$CONDA_LIB:$CONDA_LIB/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
export HF_HOME=/ephemeral/hf_cache

DATA_DIR="/ephemeral/data/tbx11k"
CKPT_BASE="/ephemeral/checkpoints"
LOG_BASE="/ephemeral/logs"
mkdir -p "$LOG_BASE" "$CKPT_BASE"

cd /home/ubuntu/Lung_cancer

# ── Step 1: Download TBX11K ───────────────────────────────────────────────────
if [ ! -f "$DATA_DIR/labels.csv" ]; then
    echo "=== Downloading TBX11K ==="
    mkdir -p /ephemeral/data/tbx11k_raw
    python3 -c "
import gdown, os
out = '/ephemeral/data/tbx11k_raw/TBX11K.zip'
if not os.path.exists(out):
    gdown.download('https://drive.google.com/uc?id=1r-oNYTPiPCOUzSjChjCIYTdkjBTugqxR', out, quiet=False)
print('Downloaded')
"
    cd /ephemeral/data/tbx11k_raw && unzip -q TBX11K.zip && cd /home/ubuntu/Lung_cancer
    python3 scripts/prepare_tbx11k.py
    echo "TBX11K ready."
else
    echo "TBX11K already prepared, skipping download."
fi

# ── Step 2: Train GRAPE (no_C_novlm) and baseline_CSR ────────────────────────
if [ ! -f "$CKPT_BASE/tbx11k_no_C_novlm/csr_stage4_best.pt" ]; then
    echo "=== Training GRAPE (GNN + Uncertainty, no VLM) ==="
    python3 train.py \
        --dataset tbx11k --data_dir "$DATA_DIR" \
        --config configs/tbx11k_config.yaml \
        --checkpoint_dir "$CKPT_BASE/tbx11k_no_C_novlm" \
        --batch_size 128 --num_workers 8 --no_vlm \
        2>&1 | tee "$LOG_BASE/tbx11k_no_C_novlm.log"
else
    echo "GRAPE checkpoint exists, skipping."
fi

if [ ! -f "$CKPT_BASE/tbx11k_baseline_CSR/csr_stage4_best.pt" ]; then
    echo "=== Training baseline CSR (no GNN, no VLM, no Uncertainty) ==="
    python3 train.py \
        --dataset tbx11k --data_dir "$DATA_DIR" \
        --config configs/tbx11k_config.yaml \
        --checkpoint_dir "$CKPT_BASE/tbx11k_baseline_CSR" \
        --batch_size 128 --num_workers 8 --no_gnn --no_vlm --no_uncertainty \
        2>&1 | tee "$LOG_BASE/tbx11k_baseline_CSR.log"
else
    echo "Baseline CSR checkpoint exists, skipping."
fi

# ── Step 3: C7 — Module B Safety Evaluation ──────────────────────────────────
echo "=== C7: Module B safety evaluation ==="
python3 scripts/c7_safety_eval.py 2>&1 | tee "$LOG_BASE/c7_safety_eval.log"

# ── Step 4: C8 — Module C Zero-Shot Leave-One-Out (TBX11K K=3 proxy) ─────────
echo "=== C8: Module C zero-shot (TBX11K K=3 proxy) ==="
python3 scripts/c8_zeroshot_eval.py 2>&1 | tee "$LOG_BASE/c8_zeroshot_eval.log"

# ── Step 5: N3 — Hyperparameter Sensitivity ───────────────────────────────────
echo "=== N3: Hyperparameter sensitivity ==="
python3 scripts/n3_sensitivity.py 2>&1 | tee "$LOG_BASE/n3_sensitivity.log"

echo ""
echo "=== ALL EXPERIMENTS DONE. Results in /ephemeral/logs/ ==="
