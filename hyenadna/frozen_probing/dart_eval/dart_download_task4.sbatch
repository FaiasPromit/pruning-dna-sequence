#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -J dart_download_task4
#SBATCH -p YOUR_CPU_PARTITION
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Download the DART-Eval Task 4 data once (see download_dart_task4.py for what is and is
# not downloaded). The files go to $DART_DIR/work:
#     refs/                                       genome FASTA
#     task_4_chromatin_activity/processed_data/
#         cell_line_expanded_peaks/   <cell>_peaks.bed, <cell>_nonpeaks.bed
#         cell_line_idr_peaks/        <cell>.bed
#         bigwigs/                    <cell>_unstranded.bw   (the regression target)
#
# The download needs a Synapse account and a personal access token (Synapse web site:
# Account Settings, Personal Access Tokens). Keep the token out of this file; set it in
# your shell and pass it to the job:
#     export SYNAPSE_AUTH_TOKEN=<your token>
#     mkdir -p logs
#     sbatch --export=ALL dart_download_task4.sbatch

if [ -z "$SYNAPSE_AUTH_TOKEN" ]; then
    echo "ERROR: SYNAPSE_AUTH_TOKEN is not set (see the top of this file)"
    exit 1
fi

DART_DIR=/path/to/DART-Eval
DART_ENV=/path/to/envs/dart
export DART_WORK_DIR=$DART_DIR/work
export PYTHONPATH=$DART_DIR/src:$PYTHONPATH
mkdir -p "$DART_WORK_DIR"

cd $SLURM_SUBMIT_DIR

module purge
module load gcc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$DART_ENV"

python download_dart_task4.py

echo ""
echo "Downloaded folders:"
for d in "refs" \
         "task_4_chromatin_activity/processed_data/cell_line_expanded_peaks" \
         "task_4_chromatin_activity/processed_data/cell_line_idr_peaks" \
         "task_4_chromatin_activity/processed_data/bigwigs"; do
    p="$DART_WORK_DIR/$d"
    if [ -d "$p" ]; then
        echo "  [ok]      $d  ($(ls "$p" | wc -l) files)"
    else
        echo "  [missing] $d"
    fi
done
echo "Total size: $(du -sh "$DART_WORK_DIR" 2>/dev/null | cut -f1)"
echo "done $(date)"
