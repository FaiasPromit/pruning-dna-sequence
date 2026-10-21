#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -J pretrain_hyenadna
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -N 12
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH -t 72:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# HyenaDNA pretraining on a SLURM cluster (12 nodes x 2 A100 80GB GPUs = 24 GPUs, as in
# the paper): Hyena on HY-HRG (1,024 bp windows) and Hyena-16k on HY16k-HRG (16,384 bp).
#
# Submit from hyenadna/pretraining/ with the script and the config of the run you want:
#     mkdir -p logs
#     sbatch -J pretrain_HY-HRG_SP_20% pretrain_hyenadna.sbatch run_clm.py configs/pretrain_HY-HRG_SP_20%.json
#
# configs/ has one file per Hyena and Hyena-16k pretraining run in the paper. Which script
# runs which variant:
#     run_clm.py              Unpruned, SP_*, RP_20% (and every HY16k-HRG run)
#     run_clm_perplexity.py   Perp_*
#     run_clm_bpas.py         BPAS_*
#
# Every run saves its final model in <output_dir>/final_model, including the B-PAS runs
# that stop early because every batch has been pruned.

# A job can occasionally stop early, for example while rank 0 is still tokenizing the
# corpus for the first time and the other ranks are waiting. Submitting the same command
# again is enough: the tokenized corpus is reused from the cache, and training resumes
# from the latest checkpoint in output_dir.

SCRIPT=$1
CONFIG=$2
RUN_NAME=$(basename "$CONFIG" .json)

# Hugging Face cache for this run. Put it on storage that every node can see,
# because rank 0 tokenizes the corpus once and the other ranks read the cached result.
CACHE_DIR=/path/to/cache/$RUN_NAME
mkdir -p $CACHE_DIR/hf_home

module purge
module load gcc
module load cuda/12.2.1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/envs/dnabert2

cd $SLURM_SUBMIT_DIR

export HF_HOME=$CACHE_DIR/hf_home
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_DISABLE_MP=1

# Settings we used for stable multi-node runs on our cluster; other clusters may not need them.
export TORCH_DIST_INIT_TIMEOUT=7200
export GLOO_SOCKET_RCVTIMEO=7200000
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=^docker0,lo

echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "Script: $SCRIPT"
echo "Config: $CONFIG"
nvidia-smi

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500
unset SLURM_CPUS_PER_TASK

# --nnodes and --nproc_per_node must match -N and --gpus-per-node above.
srun torchrun \
    --nnodes=12 \
    --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $SCRIPT $CONFIG
