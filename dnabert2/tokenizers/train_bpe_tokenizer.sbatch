#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -J train_bpe_tokenizer_DB2-MSG
#SBATCH -p YOUR_LARGE_MEMORY_PARTITION
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=2000G
#SBATCH --time=72:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Trains the 4096-token BPE tokenizer for DB2 on MSG. BPE training reads the whole
# corpus into memory, so we ran it on a single large-memory node (2 TB requested).
#
# Submit from dnabert2/tokenizers/:
#     mkdir -p logs
#     sbatch train_bpe_tokenizer.sbatch
#
# For the DB2-HRG tokenizer, set CORPUS_DIR=/path/to/HRG_corpus/DB2-HRG (written by
# data_preparation/prepare_hrg.py) and OUTDIR=DB2-HRG_retrained.
#
# The tokenizers used in the paper are already in DB2-MSG/ and DB2-HRG/, so this
# step is only needed to reproduce them. Output goes to a separate folder so the
# released files are never overwritten.

CORPUS_DIR=/path/to/MSG_corpus
OUTDIR=DB2-MSG_retrained
VOCAB_SIZE=4096
MIN_FREQ=2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/envs/dnabert2

cd $SLURM_SUBMIT_DIR
mkdir -p "$OUTDIR"

echo "Node: $SLURM_NODELIST"

python train_bpe_tokenizer.py \
  --corpus-files $CORPUS_DIR/train.txt $CORPUS_DIR/dev.txt \
  --vocab-size "$VOCAB_SIZE" \
  --min-frequency "$MIN_FREQ" \
  --save-path "$OUTDIR/tokenizer.json"

# AutoTokenizer also needs tokenizer_config.json next to tokenizer.json.
# It is the same for every DB2 tokenizer, so we copy the released one.
cp DB2-MSG/tokenizer_config.json "$OUTDIR/"
