#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -J dart_task4
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 72:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# DART-Eval Task 4 (D-T4, chromatin activity regression) with frozen probing, for one
# pretrained Hyena-16k model and one cell line (K562 in the paper):
#     1. extract frozen embeddings of the peak, non-peak and IDR regions
#     2. train the DART-Eval regression probe on them (the pretrained model is not updated)
#     3. evaluate on the test chromosomes; the paper reports test_spearman_all, the Spearman
#        correlation over the peak and non-peak test regions together
#
# Setup (DART-Eval environment, copying the files in files_for_dart_repo, downloading the data) is in
# ../README.md. Then submit from hyenadna/frozen_probing/dart_eval/ with the model folder
# and a name for the run:
#     mkdir -p logs
#     sbatch -J dart_task4_HY16k-HRG_SP_20% dart_task4.sbatch \
#         /path/to/outputs/pretrain_HY16k-HRG_SP_20%/final_model HY16k-HRG_SP_20%
#
# The run name keys every output, so different models never overwrite each other:
#     embeddings  $DART_WORK_DIR/task_4_chromatin_activity/embeddings/<run name>/
#     probe       .../supervised_models/probed/<run name>/<cell line>/v1/
#     metrics     .../supervised_model_outputs/probed/<run name>/<cell line>/eval_test.json
# Embeddings that already exist are reused, so a rerun after a time-out skips that step.

MODEL_DIR=$1
RUN_TAG=$2
CELL_LINE=${3:-K562}
if [ -z "$MODEL_DIR" ] || [ -z "$RUN_TAG" ]; then
    echo "Usage: sbatch dart_task4.sbatch <pretrained model folder> <run name> [cell line]"
    exit 1
fi

DART_DIR=/path/to/DART-Eval
DART_ENV=/path/to/envs/dart

export HYENA16K_RUN_NAME=$RUN_TAG
export HYENA16K_MODEL_DIR=$(cd "$MODEL_DIR" && pwd)
export HYENA16K_CODE_DIR=$(cd "$SLURM_SUBMIT_DIR/../../model_code" && pwd)
export HYENA16K_EMB_BATCH=256          # lower to 128 or 64 if embedding runs out of GPU memory
export DART_WORK_DIR=$DART_DIR/work
export PYTHONPATH=$DART_DIR/src:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

module purge
module load gcc
module load cuda/12.2.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$DART_ENV"

cd "$DART_DIR"
BASE=dnalm_bench.task_2_5_single.experiments.task_4_chromatin_activity
EMB_DIR="$DART_WORK_DIR/task_4_chromatin_activity/embeddings/$HYENA16K_RUN_NAME"

echo "DART-Eval Task 4 | run=$HYENA16K_RUN_NAME | cell line=$CELL_LINE | $(date)"
echo "Model folder: $HYENA16K_MODEL_DIR"
nvidia-smi

if [ ! -f "$HYENA16K_MODEL_DIR/config.json" ]; then
    echo "ERROR: $HYENA16K_MODEL_DIR/config.json not found"; exit 1
fi
if [ ! -d "$DART_WORK_DIR/task_4_chromatin_activity/processed_data" ]; then
    echo "ERROR: processed_data not found; run dart_download_task4.sbatch first"; exit 1
fi
for f in hyena16k_hrg_embeddings.py \
         experiments/task_4_chromatin_activity/extract_embeddings/hyena16k_hrg.py \
         experiments/task_4_chromatin_activity/train/hyena16k_hrg.py \
         experiments/task_4_chromatin_activity/eval_probing/hyena16k_hrg.py; do
    if [ ! -f "$DART_DIR/src/dnalm_bench/task_2_5_single/$f" ]; then
        echo "ERROR: $DART_DIR/src/dnalm_bench/task_2_5_single/$f is missing;"
        echo "       copy the files in files_for_dart_repo into DART-Eval first (see ../README.md)"
        exit 1
    fi
done

# 1. Frozen embeddings
for CATEGORY in peaks nonpeaks idr; do
    OUT="$EMB_DIR/${CELL_LINE}_${CATEGORY}.h5"
    if [ -f "$OUT" ]; then
        echo "[1/3] $CATEGORY embeddings already exist, skipping"
    else
        echo "[1/3] Extracting $CATEGORY embeddings"
        python -m $BASE.extract_embeddings.hyena16k_hrg "$CELL_LINE" "$CATEGORY" || exit 1
    fi
done

# 2. Probe training
echo "[2/3] Training the regression probe (150 epochs, pretrained model frozen)"
python -m $BASE.train.hyena16k_hrg "$CELL_LINE" || exit 1

# 3. Evaluation
echo "[3/3] Evaluating on the test chromosomes"
python -m $BASE.eval_probing.hyena16k_hrg "$CELL_LINE" test

METRICS_JSON="$DART_WORK_DIR/task_4_chromatin_activity/supervised_model_outputs/probed/$HYENA16K_RUN_NAME/$CELL_LINE/eval_test.json"
echo ""
if [ -f "$METRICS_JSON" ]; then
    python3 -c "import json, sys; print('test_spearman_all (reported in the paper):', json.load(open(sys.argv[1]))['test_spearman_all'])" "$METRICS_JSON"
    echo "All metrics:"
    cat "$METRICS_JSON"
else
    echo "No metrics file at $METRICS_JSON; check the evaluation log."
fi
echo ""
echo "done $(date)"
