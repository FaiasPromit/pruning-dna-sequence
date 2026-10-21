#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -J bend_gene_finding
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 72:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# BEND gene finding (B-GF) with frozen probing, for one pretrained Hyena-16k model:
#     1. compute frozen per-nucleotide embeddings of the gene finding data with the model
#     2. train and test BEND's convolutional probe on them (9 classes, MCC); the
#        pretrained model is not updated
#
# Setup (BEND environment, copying files_for_bend_repo, downloading the data, and adding
# each model to BEND's config files) is in ../README.md. Every model is a block in
# BEND/conf/embedding/embed.yaml, and its name is the only argument:
#     mkdir -p logs
#     sbatch -J bend_gene_finding_HY16k-HRG_SP_20 bend_gene_finding.sbatch HY16k-HRG_SP_20
#
# The name keys every output, so different models never overwrite each other:
#     embeddings  BEND/data/gene_finding/<name>/
#     probe       BEND/downstream_tasks/gene_finding/<name>/
# Embeddings that already exist are reused, so a rerun after a time-out skips that step.

EMBEDDER_NAME=$1
if [ -z "$EMBEDDER_NAME" ]; then
    echo "Usage: sbatch bend_gene_finding.sbatch <embedder name in embed.yaml>"
    exit 1
fi

TASK=gene_finding
BEND_DIR=/path/to/BEND
BEND_ENV=/path/to/envs/bend

module purge
module load gcc
module load cuda/12.2.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$BEND_ENV"

export TOKENIZERS_PARALLELISM=false

cd "$BEND_DIR"
echo "BEND $TASK | embedder=$EMBEDDER_NAME | $(date)"
nvidia-smi

if [ ! -f "data/${TASK}/${TASK}.bed" ]; then
    echo "ERROR: data/${TASK}/${TASK}.bed not found; download the BEND data first (../README.md)"
    exit 1
fi
if ! grep -q "^${EMBEDDER_NAME}:" conf/embedding/embed.yaml; then
    echo "ERROR: no '${EMBEDDER_NAME}:' block in conf/embedding/embed.yaml (see ../README.md)"
    exit 1
fi
if ! grep -q "^${EMBEDDER_NAME} *:" conf/datadims/embedding_dims.yaml; then
    echo "ERROR: no '${EMBEDDER_NAME}' line in conf/datadims/embedding_dims.yaml (see ../README.md)"
    exit 1
fi

# 1. Frozen per-nucleotide embeddings
EMB_DIR="data/${TASK}/${EMBEDDER_NAME}"
if [ ! -d "$EMB_DIR" ] || [ -z "$(ls -A "$EMB_DIR" 2>/dev/null)" ]; then
    echo "[1/2] Computing embeddings with $EMBEDDER_NAME"
    python scripts/precompute_embeddings.py model="$EMBEDDER_NAME" task="$TASK" || exit 1
else
    echo "[1/2] Embeddings already exist in $EMB_DIR, skipping"
fi

# 2. Probe training and testing
echo "[2/2] Training the probe on the frozen embeddings (9 classes, MCC)"
python scripts/train_on_task.py --config-name "$TASK" embedder="$EMBEDDER_NAME" || exit 1

# Test MCC
echo ""
METRICS_CSV="downstream_tasks/${TASK}/${EMBEDDER_NAME}/best_model_metrics.csv"
if [ -f "$METRICS_CSV" ]; then
    python3 - "$METRICS_CSV" "$EMBEDDER_NAME" << 'PYEOF'
import sys, pandas as pd
csv_path, name = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv_path)
test_cols = [c for c in df.columns if c.lower().startswith("test")]
mcc_cols = [c for c in test_cols if "mcc" in c.lower()]
col = mcc_cols[0] if mcc_cols else (test_cols[-1] if test_cols else None)
if col:
    print(f"  {name:<22} | {col} = {df[col].values[0]:.4f}")
else:
    print(f"  {name}: no 'test*' column in {csv_path}; columns={list(df.columns)}")
PYEOF
else
    echo "  No metrics file at $METRICS_CSV; check the probe training log."
fi
echo "done $(date)"
