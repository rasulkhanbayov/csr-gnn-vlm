#!/bin/bash
# Run all ablation experiments on A100
# Results are saved to /ephemeral/checkpoints/ablation_*/

BASE_ARGS="--dataset nih --data_dir /ephemeral/data/nih_cxr14 \
           --config configs/nih_config.yaml \
           --batch_size 128 --num_workers 8 \
           --num_prototypes 100 --proto_dim 256 --image_size 224"

export HF_HOME=/ephemeral/hf_cache

run_experiment() {
    local name=$1
    local extra_args=$2
    local ckpt_dir="/ephemeral/checkpoints/ablation_${name}"
    echo ""
    echo "=================================================="
    echo " ABLATION: $name"
    echo "=================================================="
    python3 train.py $BASE_ARGS --checkpoint_dir $ckpt_dir $extra_args 2>&1 | \
        tee /ephemeral/checkpoints/ablation_${name}.log | \
        grep -E "Epoch|Stage|F1|Pointing|complete|EVAL"
    echo "--- $name DONE ---"
}

# Full model (A+B+C) — already run, just re-run on real labels
run_experiment "full_ABC"       ""

# Ablation: no GNN (linear task head)
run_experiment "no_A_linear"    "--no_gnn"

# Ablation: no VLM alignment
run_experiment "no_C_novlm"     "--no_vlm"

# Ablation: baseline (no GNN, no VLM — closest to original CSR)
run_experiment "baseline_CSR"   "--no_gnn --no_vlm --no_uncertainty"

echo ""
echo "=================================================="
echo " ALL ABLATIONS COMPLETE"
echo "=================================================="
echo "Results:"
for name in full_ABC no_A_linear no_C_novlm baseline_CSR; do
    f1=$(grep "Macro F1:" /ephemeral/checkpoints/ablation_${name}.log 2>/dev/null | tail -1)
    echo "  $name: $f1"
done
