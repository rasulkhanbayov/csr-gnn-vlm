#!/bin/bash
# Monitor training progress + GPU + NIH download
# Usage: bash scripts/monitor.sh

LOGFILE="/tmp/claude-1000/-home-ubuntu-Lung-cancer/c37ff7e8-6aa8-4d8a-aa86-3a56278de7ed/tasks/bjoja5ij3.output"
DLFILE="/tmp/claude-1000/-home-ubuntu-Lung-cancer/c37ff7e8-6aa8-4d8a-aa86-3a56278de7ed/tasks/buxkic8w8.output"

echo "=========================================="
echo " CSR++ Training Monitor"
echo " $(date)"
echo "=========================================="

echo ""
echo "[GPU]"
nvidia-smi | grep -E "Temp|MiB|Util" | head -3

echo ""
echo "[NIH Download]"
echo "  Images saved: $(ls /ephemeral/data/nih_cxr14/images/ 2>/dev/null | wc -l) / 20000"
tail -2 "$DLFILE" 2>/dev/null | grep -v "^$"

echo ""
echo "[Training - last 20 lines]"
tail -20 "$LOGFILE" 2>/dev/null

echo ""
echo "[Checkpoints]"
ls -lh /ephemeral/checkpoints/*.pt 2>/dev/null || echo "  None yet"

echo ""
echo "[Disk]"
df -h /dev/vda1 /dev/vdb | tail -3
