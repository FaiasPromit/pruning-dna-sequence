#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -J pretrain_nt
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -N 12
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH -t 72:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# NT-500M pretraining on NT-HRG on a SLURM cluster
# (12 nodes x 2 A100 80GB GPUs = 24 GPUs, as in the paper).
#
# Run data_preparation/prepare_hrg.py once first; it writes NT-HRG/train.txt and dev.txt.
# Then submit from nucleotide_transformer/pretraining/ with the script and the config of
# the run you want:
#     mkdir -p logs
#     sbatch -J pretrain_NT-HRG_SP_20% pretrain_nt.sbatch train_nt.py configs/pretrain_NT-HRG_SP_20%.json
#
# configs/ has one file per NT-500M pretraining run in the paper. Which script runs which variant:
#     train_nt.py        Unpruned, SP_*
#     train_nt_bpas.py   BPAS_*
#
# The final model is saved in <output_dir>/epoch_99_checkpoint. A B-PAS run that prunes
# every batch before epoch 99 stops there, and its final model is the last checkpoint-*
# folder in output_dir.
#
# If a job stops early, submitting the same command again is enough: training resumes
# from the latest checkpoint in output_dir. At the very end, torchrun may print warnings
# that a node "failed to shutdown the rendezvous"; they come after training has finished
# and can be ignored.

SCRIPT=$1
CONFIG=$2

module purge
module load gcc
module load cuda/12.2.1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/envs/dnabert2

cd $SLURM_SUBMIT_DIR

# Settings we used for stable multi-node runs on our cluster; other clusters may not need them.
export TORCH_DIST_INIT_TIMEOUT=7200
export GLOO_SOCKET_RCVTIMEO=7200000
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
