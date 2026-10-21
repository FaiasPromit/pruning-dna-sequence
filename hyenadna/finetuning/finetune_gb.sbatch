#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -J finetune_gb
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH --array=0-6
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 72:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# Full fine-tuning of one pretrained Hyena model on the 7 Genomic Benchmarks (GB) tasks
# used in the paper, with the settings in Tables 10 and 11 (the HyenaDNA fine-tuning recipe:
# learning rate 6e-4, weight decay 0.1, cosine schedule with warmup over 1% of the steps,
# batch size 128, 20 epochs).
#
# Each task is one SLURM array task on one GPU. The maximum length of a task is the 99th
# percentile of its sequence lengths plus 2 special tokens, and the best checkpoint is the
# one with the lowest validation loss. When all 7 tasks have finished, master_summary.txt
# and master_summary.csv list the test score of every task and the mean over the 7 tasks.
#
# Run data_preparation/prepare_genomic_benchmarks.py once first; it writes the
# train/dev/test CSV files into GB_ROOT. Then submit from hyenadna/finetuning/ with the
# pretrained model folder and a name for the run:
#     mkdir -p logs
#     sbatch -J finetune_gb_HY-HRG_SP_20% finetune_gb.sbatch \
#         /path/to/outputs/pretrain_HY-HRG_SP_20%/final_model HY-HRG_SP_20%
#
# For the Full pruned baseline (fine-tuning with no pretraining at all), add random_init
# as a third argument. The model folder of any HY-HRG run is then used only for its
# config, and its weights are not loaded:
#     sbatch -J finetune_gb_HY-HRG_Full_pruned finetune_gb.sbatch \
#         /path/to/outputs/pretrain_HY-HRG_Unpruned/final_model HY-HRG_Full_pruned random_init
#
# To rerun one task, pass its number from TASK_NAMES below. The summary is rebuilt when it
# finishes:
#     sbatch --array=4 -J finetune_gb_HY-HRG_SP_20% finetune_gb.sbatch <model folder> <run name>

MODEL_DIR=$1
RUN_TAG=$2
INIT_MODE=${3:-pretrained}
if [ -z "$MODEL_DIR" ] || [ -z "$RUN_TAG" ]; then
    echo "Usage: sbatch finetune_gb.sbatch <pretrained model folder> <run name> [random_init]"
    exit 1
fi
if [ "$INIT_MODE" = "random_init" ]; then
    INIT_ARGS="--random_init True"
elif [ "$INIT_MODE" = "pretrained" ]; then
    INIT_ARGS=""
else
    echo "ERROR: the optional third argument can only be random_init (got '$INIT_MODE')"
    exit 1
fi

GB_ROOT=/path/to/Genomic_Benchmarks
OUT_ROOT=/path/to/outputs/finetune_GB/$RUN_TAG
CACHE_DIR=/path/to/cache/finetune/${RUN_TAG}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$OUT_ROOT" "$CACHE_DIR"

cd $SLURM_SUBMIT_DIR
TRAIN_PY=train_gb.py

module purge
module load gcc
module load cuda/12.2.1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/envs/dnabert2

# Everything is loaded from local files, so Hugging Face is kept offline
export HF_HOME=$CACHE_DIR
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# The model folder must hold config.json and, unless random_init is used, the weights
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: $MODEL_DIR/config.json not found"
    exit 1
fi
if [ -z "$INIT_ARGS" ] && [ ! -f "$MODEL_DIR/pytorch_model.bin" ] && ! ls "$MODEL_DIR"/*.safetensors > /dev/null 2>&1; then
    echo "ERROR: no pytorch_model.bin or .safetensors file in $MODEL_DIR"
    exit 1
fi

echo "Job ID: $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID, node $(hostname)"
echo "Model folder: $MODEL_DIR ($INIT_MODE weights)"
echo "Output folder: $OUT_ROOT"
nvidia-smi

# GB tasks, one per array task
TASK_NAMES=(
    demo_coding_vs_intergenomic_seqs   # 0
    demo_human_or_worm                 # 1
    human_enhancers_cohn               # 2
    human_enhancers_ensembl            # 3
    human_ensembl_regulatory           # 4
    human_nontata_promoters            # 5
    human_ocr_ensembl                  # 6
)

# Shared by all tasks (Table 10)
LR=6e-4
WEIGHT_DECAY=0.1
EPOCHS=20
TRAIN_BS=128
EVAL_BS=128
GRAD_ACC=1
WARMUP_RATIO=0.01
MAX_GRAD_NORM=1.0
MAX_LEN_PERCENTILE=99
DATALOADER_NW=4

if [ -z "$SLURM_ARRAY_TASK_ID" ] || [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -gt 6 ]; then
    echo "ERROR: submit with sbatch as an array job (--array=0-6)"
    exit 1
fi
IDX=$SLURM_ARRAY_TASK_ID
TASK=${TASK_NAMES[$IDX]}
DATA_DIR=$GB_ROOT/$TASK
RUN_NAME=hyenadna_$TASK
OUT_DIR=$OUT_ROOT/$TASK/$RUN_NAME
CSV=$OUT_ROOT/$TASK/training_summary.csv
mkdir -p "$OUT_ROOT/$TASK"
echo "Folder,Accuracy,MCC,F1,Precision,Recall" > "$CSV"
echo "[$TASK] $DATA_DIR ($EPOCHS epochs)"

if [ ! -f "$DATA_DIR/train.csv" ] || [ ! -f "$DATA_DIR/dev.csv" ] || [ ! -f "$DATA_DIR/test.csv" ]; then
    echo "ERROR: train.csv, dev.csv or test.csv missing in $DATA_DIR. Run data_preparation/prepare_genomic_benchmarks.py first."
    exit 1
fi

# Read the test metrics from eval_results.json and add a row to the task CSV
#   $1 = eval_results.json   $2 = task name   $3 = task CSV
extract_and_append() {
    local RESULTS_JSON=$1
    local FOLDER=$2
    local CSV_FILE=$3
    if [ -f "$RESULTS_JSON" ]; then
        local METRICS
        METRICS=$(python3 <<PYEOF
import json
try:
    with open("$RESULTS_JSON", "r") as f:
        d = json.load(f)
    print(f"{d.get('eval_accuracy',0):.4f},{d.get('eval_matthews_correlation',0):.4f},{d.get('eval_f1',0):.4f},{d.get('eval_precision',0):.4f},{d.get('eval_recall',0):.4f}")
except Exception:
    print("0.0000,0.0000,0.0000,0.0000,0.0000")
PYEOF
)
        echo "$FOLDER,$METRICS" >> "$CSV_FILE"
        echo "    [ok] $FOLDER -> $METRICS"
    else
        echo "    [missing] $FOLDER: $RESULTS_JSON not found"
        echo "$FOLDER,0.0000,0.0000,0.0000,0.0000,0.0000" >> "$CSV_FILE"
    fi
}

echo "[$TASK] started $(date)"
python "$TRAIN_PY" \
    --model_name_or_path "$MODEL_DIR" \
    $INIT_ARGS \
    --data_path "$DATA_DIR" \
    --kmer -1 \
    --run_name "$RUN_NAME" \
    --model_max_length -1 \
    --max_length_percentile $MAX_LEN_PERCENTILE \
    --per_device_train_batch_size $TRAIN_BS \
    --per_device_eval_batch_size $EVAL_BS \
    --gradient_accumulation_steps $GRAD_ACC \
    --learning_rate $LR \
    --weight_decay $WEIGHT_DECAY \
    --num_train_epochs $EPOCHS \
    --max_steps_cap 0 \
    --lr_scheduler_type cosine \
    --warmup_steps 0 \
    --warmup_ratio $WARMUP_RATIO \
    --max_grad_norm $MAX_GRAD_NORM \
    --bf16 \
    --output_dir "$OUT_DIR" \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --logging_steps 100 \
    --overwrite_output_dir \
    --log_level info \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --use_mcc_selection False \
    --dataloader_num_workers $DATALOADER_NW \
    --dataloader_pin_memory True \
    --optim adamw_torch \
    --seed 42

if [ $? -eq 0 ]; then
    extract_and_append "$OUT_DIR/results/$RUN_NAME/eval_results.json" "$TASK" "$CSV"
else
    echo "[$TASK] ERROR"
    echo "$TASK,FAIL,FAIL,FAIL,FAIL,FAIL" >> "$CSV"
fi
echo "[$TASK] done at $(date)"

# Summary of all tasks. Each task leaves a marker when it finishes; once all 7 markers
# exist, the task that finishes gets the lock and writes master_summary.txt and .csv.
# The lock is released afterwards, so rerunning a single task rebuilds the summary.
DONE_DIR=$OUT_ROOT/_done
LOCK_DIR=$OUT_ROOT/_summary_lock
mkdir -p "$DONE_DIR"
touch "$DONE_DIR/done_$IDX"
NUM_DONE=$(ls "$DONE_DIR" | wc -l)
echo "[$TASK] tasks finished: $NUM_DONE of 7"

if [ "$NUM_DONE" -ge 7 ] && mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "[$TASK] writing the master summary"
    OUT_ROOT="$OUT_ROOT" RUN_TAG="$RUN_TAG" MODEL_DIR="$MODEL_DIR" \
    TASK_ORDER="${TASK_NAMES[*]}" \
    python3 <<'PYEOF'
import csv, os, datetime

out_root = os.environ["OUT_ROOT"]
tasks = os.environ["TASK_ORDER"].split()

lines = []
lines.append(f"GB SUMMARY   run={os.environ['RUN_TAG']}   {datetime.datetime.now().isoformat(timespec='seconds')}")
lines.append(f"Model folder: {os.environ['MODEL_DIR']}")
lines.append("=" * 100)
lines.append(f"{'Task':<34} | {'Acc':<8} | {'MCC':<8} | {'F1':<8} | {'Prec':<8} | {'Rec':<8}")
lines.append("-" * 100)

per_task = {}
for t in tasks:
    csv_path = os.path.join(out_root, t, "training_summary.csv")
    if not os.path.exists(csv_path):
        lines.append(f"{t:<34} | [CSV not found]")
        continue
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader, None)
        rows = [r for r in reader if len(r) >= 6]
    if not rows:
        continue
    r = rows[0]   # one row per task
    try:
        acc, mcc, f1, pr, rc = [float(r[i]) for i in range(1, 6)]
    except ValueError:
        acc = mcc = f1 = pr = rc = 0.0
    per_task[t] = (acc, mcc, f1, pr, rc)
    lines.append(f"{t:<34} | {acc:<8.4f} | {mcc:<8.4f} | {f1:<8.4f} | {pr:<8.4f} | {rc:<8.4f}")

# Mean over the tasks (the GB score in the paper is the mean MCC)
if per_task:
    n = len(per_task)
    means = [sum(v[i] for v in per_task.values()) / n for i in range(5)]
    lines.append("-" * 100)
    lines.append(f"{'GB (mean of ' + str(n) + ' tasks)':<34} | {means[0]:<8.4f} | {means[1]:<8.4f} | {means[2]:<8.4f} | {means[3]:<8.4f} | {means[4]:<8.4f}")

with open(os.path.join(out_root, "master_summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(out_root, "master_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Task", "Accuracy", "MCC", "F1", "Precision", "Recall"])
    for t in tasks:
        if t in per_task:
            w.writerow([t] + [f"{x:.4f}" for x in per_task[t]])
    if per_task:
        w.writerow([])
        w.writerow([f"GB mean ({len(per_task)} tasks)"] + [f"{x:.4f}" for x in means])
print("\n".join(lines))
PYEOF
    rmdir "$LOCK_DIR"
    echo "Summary written to $OUT_ROOT/master_summary.txt and master_summary.csv"
fi
