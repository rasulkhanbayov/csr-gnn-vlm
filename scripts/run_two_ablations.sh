#!/bin/bash
# Run only GRAPE (no_C_novlm) and baseline_CSR — needed for qualitative figure
CONDA_LIB=/opt/anaconda3/2024.02-1/conda_envs/ml_dl_gpu_base/lib
export LD_LIBRARY_PATH=$CONDA_LIB:$CONDA_LIB/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
export HF_HOME=/ephemeral/hf_cache

DATA_DIR="/ephemeral/data/tbx11k"
CKPT_BASE="/ephemeral/checkpoints"
LOG_BASE="/ephemeral/logs"
mkdir -p "$LOG_BASE"

cd /home/ubuntu/Lung_cancer

echo "=== GRAPE (no_C_novlm): GNN + Uncertainty ==="
python3 train.py \
    --dataset tbx11k --data_dir "$DATA_DIR" \
    --config configs/tbx11k_config.yaml \
    --checkpoint_dir "$CKPT_BASE/tbx11k_no_C_novlm" \
    --batch_size 128 --num_workers 8 \
    --no_vlm \
    2>&1 | tee "$LOG_BASE/tbx11k_no_C_novlm.log"

echo "=== baseline_CSR: no GNN, no VLM, no Uncertainty ==="
python3 train.py \
    --dataset tbx11k --data_dir "$DATA_DIR" \
    --config configs/tbx11k_config.yaml \
    --checkpoint_dir "$CKPT_BASE/tbx11k_baseline_CSR" \
    --batch_size 128 --num_workers 8 \
    --no_gnn --no_vlm --no_uncertainty \
    2>&1 | tee "$LOG_BASE/tbx11k_baseline_CSR.log"

echo "Both done."
