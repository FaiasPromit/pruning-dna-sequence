#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -J prepare_hrg
#SBATCH -p YOUR_CPU_PARTITION
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Download the human reference genome and write the four HRG pretraining corpora,
# DB2-HRG, NT-HRG, HY-HRG and HY16k-HRG (see prepare_hrg.py). Submit from data_preparation/:
#     mkdir -p logs
#     sbatch prepare_hrg.sbatch
# The pretraining configs then use /path/to/HRG_corpus/<corpus>/train.txt and dev.txt.

OUTPUT_DIR=/path/to/HRG_corpus

module purge
module load gcc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/envs/dnabert2

cd $SLURM_SUBMIT_DIR
echo "Job ID: $SLURM_JOB_ID"
echo "Output folder: $OUTPUT_DIR"
echo "Start: $(date)"

python prepare_hrg.py --output_dir "$OUTPUT_DIR"

echo "Done: $(date)"
