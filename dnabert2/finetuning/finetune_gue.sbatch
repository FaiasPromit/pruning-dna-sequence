#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -J finetune_gue
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH --array=0-7
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# Full fine-tuning of one pretrained DNABERT-2 model on the 27 GUE tasks (every GUE task
# except the virus task), with the settings in Tables 10 and 11 of the paper.
#
# The tasks are split into 8 groups, one per SLURM array task. Each array task uses one GPU
# and fine-tunes the tasks of its group one after another. When all 8 groups have finished,
# master_summary.txt and master_summary.csv list the test score of every task and the
# average of each task group (emp, mouse, core, prom, splice, tf).
#
# Submit from dnabert2/finetuning/ with the pretrained model folder and a name for the run:
#     mkdir -p logs
#     sbatch -J finetune_gue_DB2-MSG_SP_20% finetune_gue.sbatch \
#         /path/to/outputs/pretrain_DB2-MSG_SP_20%/final_model DB2-MSG_SP_20%
#
# The model folder can be one saved by the pretraining scripts or a model downloaded from
# Hugging Face; both hold the weights, the tokenizer and the model code. GUE is used as
# downloaded from the DNABERT-2 repository (https://github.com/MAGICS-LAB/DNABERT_2).
#
# For the Full pruned baseline (fine-tuning with no pretraining at all), add random_init
# as a third argument. The model folder of any run on the same corpus is then used only
# for its config and tokenizer, and its weights are not loaded:
#     sbatch -J finetune_gue_DB2-MSG_Full_pruned finetune_gue.sbatch \
#         /path/to/outputs/pretrain_DB2-MSG_Unpruned/final_model DB2-MSG_Full_pruned random_init
#
# To rerun one group, for example after a time-out, pass its number from GROUP_NAMES below.
# The summary is rebuilt when it finishes:
#     sbatch --array=3 -J finetune_gue_DB2-MSG_SP_20% finetune_gue.sbatch <model folder> <run name>

MODEL_DIR=$1
RUN_TAG=$2
INIT_MODE=${3:-pretrained}
if [ -z "$MODEL_DIR" ] || [ -z "$RUN_TAG" ]; then
    echo "Usage: sbatch finetune_gue.sbatch <pretrained model folder> <run name> [random_init]"
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

GUE_ROOT=/path/to/GUE
OUT_ROOT=/path/to/outputs/finetune_GUE/$RUN_TAG
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

echo "Job ID: $SLURM_ARRAY_JOB_ID, group $SLURM_ARRAY_TASK_ID, node $(hostname)"
echo "Model folder: $MODEL_DIR ($INIT_MODE weights)"
echo "Output folder: $OUT_ROOT"
nvidia-smi

# GUE task groups, one per array task. An empty GROUP_FOLDERS entry means every task folder
# under that GUE folder. The promoter tasks are split so that the TATA tasks get 10 epochs.
# GROUP_MAX_LEN is the maximum number of tokens per sequence (Table 11, DB2 column).
GROUP_NAMES=(emp mouse core_all core_tata prom_all prom_tata splice tf)
GROUP_DIRS=(EMP mouse prom prom prom prom splice tf)
GROUP_FOLDERS=(""
               ""
               "prom_core_all prom_core_notata"
               "prom_core_tata"
               "prom_300_all prom_300_notata"
               "prom_300_tata"
               ""
               "")
GROUP_MAX_LEN=(132 47 21 27 77 82 110 29)
GROUP_EPOCHS=(3 30 4 10 4 10 5 3)

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

if [ -z "$SLURM_ARRAY_TASK_ID" ] || [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -gt 7 ]; then
    echo "ERROR: submit with sbatch as an array job (--array=0-7)"
    exit 1
fi
IDX=$SLURM_ARRAY_TASK_ID
GROUP=${GROUP_NAMES[$IDX]}
DATA_DIR=$GUE_ROOT/${GROUP_DIRS[$IDX]}
MAX_LEN=${GROUP_MAX_LEN[$IDX]}
EPOCHS=${GROUP_EPOCHS[$IDX]}
FOLDERS=(${GROUP_FOLDERS[$IDX]})
if [ ${#FOLDERS[@]} -eq 0 ]; then
    FOLDERS=($(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d ! -name ".*" -exec basename {} \; | sort))
fi

GROUP_OUT=$OUT_ROOT/$GROUP
mkdir -p "$GROUP_OUT"
CSV=$GROUP_OUT/training_summary.csv
echo "Folder,Accuracy,MCC,F1,Precision,Recall" > "$CSV"
echo "[$GROUP] $DATA_DIR: ${FOLDERS[*]} (max length $MAX_LEN, $EPOCHS epochs)"

# Read the test metrics of one task from eval_results.json and add a row to the group CSV
#   $1 = eval_results.json   $2 = task folder name   $3 = group CSV
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

# Write the group CSV as a table with the group average, to training_summary.txt
#   $1 = group name   $2 = group output folder
build_group_summary() {
    GROUP_NAME="$1" CSV_PATH="$2/training_summary.csv" TXT_PATH="$2/training_summary.txt" \
    python3 <<'PYEOF'
import csv, os
rows = []
if os.path.exists(os.environ["CSV_PATH"]):
    with open(os.environ["CSV_PATH"]) as f:
        reader = csv.reader(f)
        next(reader, None)
        rows = [r for r in reader if len(r) >= 6]
out = [f"GROUP SUMMARY: {os.environ['GROUP_NAME']}", "=" * 88,
       f"{'Task':<22} | {'Accuracy':<10} | {'MCC':<10} | {'F1':<10} | {'Precision':<10} | {'Recall':<10}",
       "-" * 88]
cols = [[], [], [], [], []]
for r in rows:
    out.append(f"{r[0]:<22} | {r[1]:<10} | {r[2]:<10} | {r[3]:<10} | {r[4]:<10} | {r[5]:<10}")
    try:
        vals = [float(x) for x in r[1:6]]
    except ValueError:
        continue
    for c, v in zip(cols, vals):
        c.append(v)
avg = [sum(c) / len(c) if c else 0.0 for c in cols]
out += ["-" * 88,
        f"{'AVERAGE':<22} | {avg[0]:<10.4f} | {avg[1]:<10.4f} | {avg[2]:<10.4f} | {avg[3]:<10.4f} | {avg[4]:<10.4f}"]
with open(os.environ["TXT_PATH"], "w") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
PYEOF
}

for FOLDER in "${FOLDERS[@]}"; do
    FOLDER_PATH=$DATA_DIR/$FOLDER
    if [ ! -f "$FOLDER_PATH/train.csv" ] || [ ! -f "$FOLDER_PATH/dev.csv" ] || [ ! -f "$FOLDER_PATH/test.csv" ]; then
        echo "[$GROUP] skipping $FOLDER (train.csv, dev.csv or test.csv missing)"
        continue
    fi

    RUN_NAME=dnabert2_${GROUP}_${FOLDER}
    OUT_DIR=$GROUP_OUT/$RUN_NAME
    echo "[$GROUP] $FOLDER started $(date)"

    python "$TRAIN_PY" \
        --model_name_or_path "$MODEL_DIR" \
        $INIT_ARGS \
        --data_path "$FOLDER_PATH" \
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
        --dataloader_num_workers $DATALOADER_NW \
        --dataloader_pin_memory True \
        --ddp_find_unused_parameters True \
        --optim adamw_torch \
        --ddp_backend nccl

    if [ $? -eq 0 ]; then
        extract_and_append "$OUT_DIR/results/$RUN_NAME/eval_results.json" "$FOLDER" "$CSV"
    else
        echo "[$GROUP] ERROR on $FOLDER"
        echo "$FOLDER,FAIL,FAIL,FAIL,FAIL,FAIL" >> "$CSV"
    fi
done
build_group_summary "$GROUP" "$GROUP_OUT"
echo "[$GROUP] done at $(date)"

# Summary of all groups. Each group leaves a marker when it finishes; once all 8 markers
# exist, the group that finishes gets the lock and writes master_summary.txt and .csv.
# The lock is released afterwards, so rerunning a single group rebuilds the summary.
DONE_DIR=$OUT_ROOT/_done
LOCK_DIR=$OUT_ROOT/_summary_lock
mkdir -p "$DONE_DIR"
touch "$DONE_DIR/done_$IDX"
NUM_DONE=$(ls "$DONE_DIR" | wc -l)
echo "[$GROUP] groups finished: $NUM_DONE of 8"

if [ "$NUM_DONE" -ge 8 ] && mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "[$GROUP] writing the master summary"
    OUT_ROOT="$OUT_ROOT" RUN_TAG="$RUN_TAG" MODEL_DIR="$MODEL_DIR" \
    GROUP_ORDER="${GROUP_NAMES[*]}" \
    python3 <<'PYEOF'
import csv, os, datetime

out_root = os.environ["OUT_ROOT"]
groups = os.environ["GROUP_ORDER"].split()

lines = []
lines.append(f"GUE SUMMARY   run={os.environ['RUN_TAG']}   {datetime.datetime.now().isoformat(timespec='seconds')}")
lines.append(f"Model folder: {os.environ['MODEL_DIR']}")
lines.append("=" * 90)

all_rows = []
per_group = {}   # group -> lists of task-level metrics
for g in groups:
    csv_path = os.path.join(out_root, g, "training_summary.csv")
    lines.append("")
    lines.append(f"GROUP: {g}")
    lines.append("-" * 90)
    if not os.path.exists(csv_path):
        lines.append(f"  [CSV not found: {csv_path}]")
        continue
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader, None)
        rows = [r for r in reader if len(r) >= 6]
    lines.append(f"{'Task':<25} | {'Acc':<8} | {'MCC':<8} | {'F1':<8} | {'Prec':<8} | {'Rec':<8}")
    lines.append("-" * 90)
    m = dict(accs=[], mccs=[], f1s=[], precs=[], recs=[])
    for r in rows:
        lines.append(f"{r[0]:<25} | {r[1]:<8} | {r[2]:<8} | {r[3]:<8} | {r[4]:<8} | {r[5]:<8}")
        all_rows.append([g] + r)
        try:
            vals = [float(x) for x in r[1:6]]
        except ValueError:
            continue
        for key, v in zip(m, vals):
            m[key].append(v)
    per_group[g] = m
    avg = [sum(v) / len(v) if v else 0.0 for v in m.values()]
    lines.append("-" * 90)
    lines.append(f"{'AVERAGE':<25} | {avg[0]:<8.4f} | {avg[1]:<8.4f} | {avg[2]:<8.4f} | {avg[3]:<8.4f} | {avg[4]:<8.4f}")


def mean_over(keys):
    """Mean of each metric over all task folders of the given groups."""
    pooled = dict(accs=[], mccs=[], f1s=[], precs=[], recs=[])
    for k in keys:
        for key in pooled:
            pooled[key] += per_group.get(k, {}).get(key, [])
    return tuple(sum(v) / len(v) if v else 0.0 for v in pooled.values())


# Six reported groups. core pools prom_core_all, prom_core_notata and prom_core_tata;
# prom pools prom_300_all, prom_300_notata and prom_300_tata.
reported = {
    "emp": mean_over(["emp"]),
    "mouse": mean_over(["mouse"]),
    "core": mean_over(["core_all", "core_tata"]),
    "prom": mean_over(["prom_all", "prom_tata"]),
    "splice": mean_over(["splice"]),
    "tf": mean_over(["tf"]),
}

lines.append("")
lines.append("=" * 90)
lines.append("GROUP AVERAGES")
lines.append("-" * 90)
lines.append(f"{'Group':<15} | {'Acc':<8} | {'MCC':<8} | {'F1':<8} | {'Prec':<8} | {'Rec':<8}")
lines.append("-" * 90)
for g, v in reported.items():
    lines.append(f"{g:<15} | {v[0]:<8.4f} | {v[1]:<8.4f} | {v[2]:<8.4f} | {v[3]:<8.4f} | {v[4]:<8.4f}")
grand = [sum(v[i] for v in reported.values()) / len(reported) for i in range(5)]
lines.append("-" * 90)
lines.append(f"{'GUE (mean)':<15} | {grand[0]:<8.4f} | {grand[1]:<8.4f} | {grand[2]:<8.4f} | {grand[3]:<8.4f} | {grand[4]:<8.4f}")

with open(os.path.join(out_root, "master_summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(out_root, "master_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Group", "Task", "Accuracy", "MCC", "F1", "Precision", "Recall"])
    for r in all_rows:
        w.writerow(r)
    w.writerow([])
    w.writerow(["Group", "Accuracy", "MCC", "F1", "Precision", "Recall"])
    for g, v in reported.items():
        w.writerow([g] + [f"{x:.4f}" for x in v])
print("\n".join(lines))
PYEOF
    rmdir "$LOCK_DIR"
    echo "Summary written to $OUT_ROOT/master_summary.txt and master_summary.csv"
fi
