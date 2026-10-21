#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -J finetune_nt18
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH --array=0-17
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# Full fine-tuning of one pretrained DNABERT-2 model on the 18 Nucleotide Transformer
# downstream tasks (NT-18, InstaDeepAI/nucleotide_transformer_downstream_tasks_revised),
# with the settings in Tables 10 and 11 of the paper.
#
# Each task is one SLURM array task on one GPU. The best checkpoint is the one with the
# highest validation MCC. When all 18 tasks have finished, master_summary.txt and
# master_summary.csv list the test score of every task and the average of each category
# (regulatory, splicing, chromatin).
#
# Run data_preparation/prepare_nt18.py once first; it downloads the tasks and writes the
# train/dev/test CSV files into NT18_ROOT. Then submit from dnabert2/finetuning/ with the pretrained
# model folder and a name for the run:
#     mkdir -p logs
#     sbatch -J finetune_nt18_DB2-MSG_SP_20% finetune_nt18.sbatch \
#         /path/to/outputs/pretrain_DB2-MSG_SP_20%/final_model DB2-MSG_SP_20%
#
# For the Full pruned baseline (fine-tuning with no pretraining at all), add random_init
# as a third argument. The model folder of any run on the same corpus is then used only
# for its config and tokenizer, and its weights are not loaded:
#     sbatch -J finetune_nt18_DB2-MSG_Full_pruned finetune_nt18.sbatch \
#         /path/to/outputs/pretrain_DB2-MSG_Unpruned/final_model DB2-MSG_Full_pruned random_init
#
# To rerun one task, pass its number from TASK_NAMES below. The summary is rebuilt when it
# finishes:
#     sbatch --array=5 -J finetune_nt18_DB2-MSG_SP_20% finetune_nt18.sbatch <model folder> <run name>

MODEL_DIR=$1
RUN_TAG=$2
INIT_MODE=${3:-pretrained}
if [ -z "$MODEL_DIR" ] || [ -z "$RUN_TAG" ]; then
    echo "Usage: sbatch finetune_nt18.sbatch <pretrained model folder> <run name> [random_init]"
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

NT18_ROOT=/path/to/NT18
OUT_ROOT=/path/to/outputs/finetune_NT18/$RUN_TAG
CACHE_DIR=/path/to/cache/finetune/${RUN_TAG}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$OUT_ROOT" "$CACHE_DIR"

cd $SLURM_SUBMIT_DIR
TRAIN_PY=train.py

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

# The model folder must hold config.json, the weights, the tokenizer and the model code
for f in config.json bert_layers.py configuration_bert.py; do
    if [ ! -f "$MODEL_DIR/$f" ]; then
        echo "ERROR: $MODEL_DIR/$f not found."
        echo "       Copy the files from dnabert2/model_code/ into the model folder."
        exit 1
    fi
done
# The weights are not needed for random_init
if [ -z "$INIT_ARGS" ] && [ ! -f "$MODEL_DIR/pytorch_model.bin" ] && ! ls "$MODEL_DIR"/*.safetensors > /dev/null 2>&1; then
    echo "ERROR: no pytorch_model.bin or .safetensors file in $MODEL_DIR"
    exit 1
fi

echo "Job ID: $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID, node $(hostname)"
echo "Model folder: $MODEL_DIR ($INIT_MODE weights)"
echo "Output folder: $OUT_ROOT"
nvidia-smi

# NT-18 tasks, one per array task. TASK_MAX_LEN is the maximum number of tokens per
# sequence (Table 11, DB2 column).
TASK_NAMES=(
    promoter_all            # 0
    promoter_tata           # 1
    promoter_no_tata        # 2
    enhancers               # 3
    enhancers_types         # 4
    splice_sites_all        # 5
    splice_sites_acceptors  # 6
    splice_sites_donors     # 7
    H2AFZ                   # 8
    H3K27ac                 # 9
    H3K27me3                # 10
    H3K36me3                # 11
    H3K4me1                 # 12
    H3K4me2                 # 13
    H3K4me3                 # 14
    H3K9ac                  # 15
    H3K9me3                 # 16
    H4K20me1                # 17
)
TASK_MAX_LEN=(
    100 100 100                                  # promoters (300 bp)
    128 128                                      # enhancers (400 bp)
    192 192 192                                  # splice sites (600 bp)
    256 256 256 256 256 256 256 256 256 256      # histone marks (1,000 bp)
)
TASK_EPOCHS=(
    4 10 4                                       # promoters
    4 4                                          # enhancers
    5 5 5                                        # splice sites
    3 3 3 3 3 3 3 3 3 3                          # histone marks
)

# Shared by all tasks (Table 10). Each task runs on one GPU, so the batch size is 16.
LR=3e-5
TRAIN_BS=16
EVAL_BS=32
GRAD_ACC=1
SAVE_STEPS=200
EVAL_STEPS=200
WARMUP=100
LOG_STEPS=100
DATALOADER_NW=4

if [ -z "$SLURM_ARRAY_TASK_ID" ] || [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -gt 17 ]; then
    echo "ERROR: submit with sbatch as an array job (--array=0-17)"
    exit 1
fi
IDX=$SLURM_ARRAY_TASK_ID
TASK=${TASK_NAMES[$IDX]}
MAX_LEN=${TASK_MAX_LEN[$IDX]}
EPOCHS=${TASK_EPOCHS[$IDX]}
DATA_DIR=$NT18_ROOT/$TASK
RUN_NAME=dnabert2_$TASK
OUT_DIR=$OUT_ROOT/$TASK/$RUN_NAME
CSV=$OUT_ROOT/$TASK/training_summary.csv
mkdir -p "$OUT_ROOT/$TASK"
echo "Folder,Accuracy,MCC,F1,Precision,Recall" > "$CSV"
echo "[$TASK] $DATA_DIR (max length $MAX_LEN, $EPOCHS epochs)"

if [ ! -f "$DATA_DIR/train.csv" ] || [ ! -f "$DATA_DIR/dev.csv" ] || [ ! -f "$DATA_DIR/test.csv" ]; then
    echo "ERROR: train.csv, dev.csv or test.csv missing in $DATA_DIR. Run data_preparation/prepare_nt18.py first."
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
    --model_max_length $MAX_LEN \
    --per_device_train_batch_size $TRAIN_BS \
    --per_device_eval_batch_size $EVAL_BS \
    --gradient_accumulation_steps $GRAD_ACC \
    --learning_rate $LR \
    --num_train_epochs $EPOCHS \
    --bf16 \
    --save_steps $SAVE_STEPS \
    --output_dir "$OUT_DIR" \
    --evaluation_strategy steps \
    --eval_steps $EVAL_STEPS \
    --warmup_steps $WARMUP \
    --logging_steps $LOG_STEPS \
    --overwrite_output_dir \
    --log_level info \
    --find_unused_parameters False \
    --use_mcc_selection False \
    --load_best_model_at_end True \
    --metric_for_best_model matthews_correlation \
    --greater_is_better True \
    --dataloader_num_workers $DATALOADER_NW \
    --dataloader_pin_memory True \
    --ddp_find_unused_parameters True \
    --optim adamw_torch \
    --ddp_backend nccl

if [ $? -eq 0 ]; then
    extract_and_append "$OUT_DIR/results/$RUN_NAME/eval_results.json" "$TASK" "$CSV"
else
    echo "[$TASK] ERROR"
    echo "$TASK,FAIL,FAIL,FAIL,FAIL,FAIL" >> "$CSV"
fi
echo "[$TASK] done at $(date)"

# Summary of all tasks. Each task leaves a marker when it finishes; once all 18 markers
# exist, the task that finishes gets the lock and writes master_summary.txt and .csv.
# The lock is released afterwards, so rerunning a single task rebuilds the summary.
DONE_DIR=$OUT_ROOT/_done
LOCK_DIR=$OUT_ROOT/_summary_lock
mkdir -p "$DONE_DIR"
touch "$DONE_DIR/done_$IDX"
NUM_DONE=$(ls "$DONE_DIR" | wc -l)
echo "[$TASK] tasks finished: $NUM_DONE of 18"

if [ "$NUM_DONE" -ge 18 ] && mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "[$TASK] writing the master summary"
    OUT_ROOT="$OUT_ROOT" RUN_TAG="$RUN_TAG" MODEL_DIR="$MODEL_DIR" \
    TASK_ORDER="${TASK_NAMES[*]}" \
    python3 <<'PYEOF'
import csv, os, datetime

out_root = os.environ["OUT_ROOT"]
tasks = os.environ["TASK_ORDER"].split()

# NT-18 reporting categories
CATEGORY = {
    "promoter_all": "regulatory", "promoter_tata": "regulatory", "promoter_no_tata": "regulatory",
    "enhancers": "regulatory", "enhancers_types": "regulatory",
    "splice_sites_all": "splicing", "splice_sites_acceptors": "splicing", "splice_sites_donors": "splicing",
    "H2AFZ": "chromatin", "H3K27ac": "chromatin", "H3K27me3": "chromatin", "H3K36me3": "chromatin",
    "H3K4me1": "chromatin", "H3K4me2": "chromatin", "H3K4me3": "chromatin",
    "H3K9ac": "chromatin", "H3K9me3": "chromatin", "H4K20me1": "chromatin",
}

lines = []
lines.append(f"NT-18 SUMMARY   run={os.environ['RUN_TAG']}   {datetime.datetime.now().isoformat(timespec='seconds')}")
lines.append(f"Model folder: {os.environ['MODEL_DIR']}")
lines.append("=" * 100)
lines.append(f"{'Task':<25} | {'Category':<12} | {'Acc':<8} | {'MCC':<8} | {'F1':<8} | {'Prec':<8} | {'Rec':<8}")
lines.append("-" * 100)

per_task = {}
for t in tasks:
    csv_path = os.path.join(out_root, t, "training_summary.csv")
    if not os.path.exists(csv_path):
        lines.append(f"{t:<25} | {CATEGORY[t]:<12} | [CSV not found]")
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
    lines.append(f"{t:<25} | {CATEGORY[t]:<12} | {acc:<8.4f} | {mcc:<8.4f} | {f1:<8.4f} | {pr:<8.4f} | {rc:<8.4f}")

# Mean over the tasks of each category, then the mean of the three categories
lines.append("=" * 100)
lines.append("CATEGORY AVERAGES")
lines.append("-" * 100)
lines.append(f"{'Category':<15} | {'n':<3} | {'Acc':<8} | {'MCC':<8} | {'F1':<8} | {'Prec':<8} | {'Rec':<8}")
lines.append("-" * 100)
cat_means = {}
for cat in ["regulatory", "splicing", "chromatin"]:
    vals = [per_task[t] for t in tasks if CATEGORY[t] == cat and t in per_task]
    if not vals:
        continue
    n = len(vals)
    means = [sum(v[i] for v in vals) / n for i in range(5)]
    cat_means[cat] = means
    lines.append(f"{cat:<15} | {n:<3} | {means[0]:<8.4f} | {means[1]:<8.4f} | {means[2]:<8.4f} | {means[3]:<8.4f} | {means[4]:<8.4f}")
if cat_means:
    grand = [sum(cm[i] for cm in cat_means.values()) / len(cat_means) for i in range(5)]
    lines.append("-" * 100)
    lines.append(f"{'NT-18 (mean)':<15} | {'':<3} | {grand[0]:<8.4f} | {grand[1]:<8.4f} | {grand[2]:<8.4f} | {grand[3]:<8.4f} | {grand[4]:<8.4f}")

with open(os.path.join(out_root, "master_summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(out_root, "master_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Task", "Category", "Accuracy", "MCC", "F1", "Precision", "Recall"])
    for t in tasks:
        if t in per_task:
            w.writerow([t, CATEGORY[t]] + [f"{x:.4f}" for x in per_task[t]])
    w.writerow([])
    w.writerow(["Category", "n", "Accuracy", "MCC", "F1", "Precision", "Recall"])
    for cat, m in cat_means.items():
        n = sum(1 for t in tasks if CATEGORY[t] == cat and t in per_task)
        w.writerow([cat, n] + [f"{x:.4f}" for x in m])
print("\n".join(lines))
PYEOF
    rmdir "$LOCK_DIR"
    echo "Summary written to $OUT_ROOT/master_summary.txt and master_summary.csv"
fi
