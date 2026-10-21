# Pruning DNA Sequence

This repository has the code of the paper **Pruning DNA Sequence**: the pretraining,
fine-tuning, frozen-probing and analysis scripts behind every result in the paper, with one
config file for each pretraining run.

Paper: coming soon on arXiv.
Pretrained models: coming soon on Hugging Face.

## What the method does

Sequence Pruning (SP) removes a DNA sequence from the pretraining corpus once the model can
already predict it well. While the model trains, each sequence's accuracy is read from the
forward pass the model makes anyway, so no separate scoring model and no extra pass over the
corpus are needed. At the end of every epoch, each sequence whose accuracy lies between a low
threshold and 100% is removed for the rest of the run. Variants are named after the low
threshold, so SP_20% removes a sequence once at least 20% of its predicted tokens are right.

The savings are measured with the Data Savings Index (DSI), the share of the unpruned
baseline's sequence processing that a pruned run avoids. A DSI of 50% means the run processed
half as many sequences as the baseline.

## The five pretraining settings

A setting is one model trained on one corpus. MSG is the multi-species genome corpus of
DNABERT-2, and HRG is the human reference genome (GRCh38). In this repository each setting is
known by its corpus name, which is also the prefix of its config files.

| Setting in the paper | Name here | Model | Folder | Downstream benchmarks |
|---|---|---|---|---|
| DB2 on MSG | DB2-MSG | DNABERT-2, 117M | `dnabert2/` | GUE, NT-18 |
| DB2 on HRG | DB2-HRG | DNABERT-2, 117M | `dnabert2/` | GUE, NT-18 |
| NT-500M on HRG | NT-HRG | Nucleotide Transformer, 500M | `nucleotide_transformer/` | GUE, NT-18 |
| Hyena on HRG | HY-HRG | HyenaDNA, 1,024 bp | `hyenadna/` | GB |
| Hyena-16k on HRG | HY16k-HRG | HyenaDNA, 16,384 bp | `hyenadna/` | B-GF, D-T4 |

GUE, NT-18 and GB (Genomic Benchmarks) are evaluated by full fine-tuning. B-GF (BEND gene
finding) and D-T4 (DART-Eval Task 4) are evaluated by frozen probing, where the pretrained
model only turns sequences into embeddings and a small probe is trained on top.

## Repository layout

```
pruning-dna-sequence/
  LICENSE                     Apache License 2.0
  requirements.txt            Python environment used for every run in the paper
  data_preparation/           builds the HRG corpora and the NT-18 and GB datasets
  dnabert2/
    tokenizers/               the DB2-MSG and DB2-HRG BPE tokenizers, and the script that trained them
    model_code/               DNABERT-2 model code (from DNABERT-2 and MosaicBERT)
    pretraining/              pretraining scripts, configs/ and the SLURM script
    finetuning/               GUE and NT-18 fine-tuning
  nucleotide_transformer/
    tokenizer/                the 6-mer tokenizer and its vocabulary
    pretraining/              pretraining scripts, configs/ and the SLURM script
    finetuning/               GUE and NT-18 fine-tuning
  hyenadna/
    model_code/               HyenaDNA model code and character tokenizer
    pretraining/              pretraining scripts, configs/ and the SLURM script (Hyena and Hyena-16k)
    finetuning/               Genomic Benchmarks fine-tuning
    frozen_probing/           BEND gene finding and DART-Eval Task 4, with their own README
  analysis/                   DSI computation and the scripts for Figures 1, 2 and 6
```

## Setup

Every pretraining and fine-tuning run in the paper used one conda environment with Python 3.8
and PyTorch 1.13.1 for CUDA 11.7:

```
conda create --prefix /path/to/envs/dnabert2 python=3.8 -y
conda activate /path/to/envs/dnabert2
pip install torch==1.13.1+cu117 torchaudio==0.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
python -c "import torch; print(torch.__version__)"
pip install genomic-benchmarks
```

The last line is only needed to download the Genomic Benchmarks data. Triton is not part of
this environment, so DNABERT-2 uses its PyTorch attention, as it did in all runs of the paper,
and prints a warning at start-up that it could not import Triton. The warning is expected.

BEND and DART-Eval each need their own environment. The steps are in
[`hyenadna/frozen_probing/README.md`](hyenadna/frozen_probing/README.md).

All jobs were run on a SLURM cluster, and the `.sbatch` files are the job scripts we used.
Before submitting one, replace `YOUR_ACCOUNT`, `YOUR_GPU_PARTITION` and the other `YOUR_...`
names with those of your cluster, set every `/path/to/...` to your own folders, and change the
`module load` lines if your cluster uses different modules. Each `.sbatch` file explains at
the top how to submit it.

## Data

**MSG.** Download the DNABERT-2 pretraining data from the link in Section 5 of the
[DNABERT-2 README](https://github.com/MAGICS-LAB/DNABERT_2)
([direct link](https://drive.google.com/file/d/1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f/view?usp=sharing)).
It has `train.txt` and `dev.txt`, which the DB2-MSG configs read from `/path/to/MSG_corpus/`.

**HRG.** `data_preparation/prepare_hrg.py` downloads GRCh38 from NCBI and writes the four HRG
corpora, one sequence per line, each in its own folder under `/path/to/HRG_corpus/`.
Chromosome 21 is the dev set. For HY-HRG, chromosomes 14 and X are held out in `test.txt`, as
in the HyenaDNA paper, and are not used for pretraining.

```
cd data_preparation
python prepare_hrg.py --output_dir /path/to/HRG_corpus
```

or `sbatch prepare_hrg.sbatch` on the cluster. To check the result, these are the line counts
of the files it writes:

| Corpus | Sequence length | train.txt | dev.txt |
|---|---|---|---|
| DB2-HRG | 1,000 bp, sequences with N dropped | 2,896,771 | 40,040 |
| NT-HRG | 6,100 bp windows, every 6,000 bp | 506,913 | 7,784 |
| HY-HRG | 1,024 bp | 2,713,345 | 45,615 |
| HY16k-HRG | 16,384 bp | 185,631 | 2,850 |

HY-HRG also has a `test.txt` with 256,917 lines. For comparison, the MSG `train.txt` has
32,387,832 lines and its `dev.txt` has 100,000.

**GUE.** Download GUE from the
[DNABERT-2 repository](https://github.com/MAGICS-LAB/DNABERT_2)
([direct link](https://drive.google.com/file/d/1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-/view?usp=sharing))
and set `GUE_ROOT` in the GUE fine-tuning scripts to the unzipped folder. The paper uses every
GUE task except the virus task.

**NT-18.** `data_preparation/prepare_nt18.py` downloads the 18 tasks from
[InstaDeepAI/nucleotide_transformer_downstream_tasks_revised](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised)
and sets aside 10% of each training split as the dev split:

```
python prepare_nt18.py --out /path/to/NT18
```

**Genomic Benchmarks.** `data_preparation/prepare_genomic_benchmarks.py` downloads the seven
GB tasks used in the paper and also sets aside 10% of each training split as the dev split:

```
python prepare_genomic_benchmarks.py --output_dir /path/to/Genomic_Benchmarks
```

Run these download scripts on a machine with internet access, such as a login node. The data
for BEND and DART-Eval is downloaded with the benchmarks' own tools, as described in the
frozen-probing README.

**Tokenizers.** The two DNABERT-2 BPE tokenizers used in the paper are already in
`dnabert2/tokenizers/DB2-MSG/` and `dnabert2/tokenizers/DB2-HRG/`. The script next to them is
only needed to train them again. The NT-500M vocabulary is in
`nucleotide_transformer/tokenizer/`, and the HyenaDNA character tokenizer is part of
`hyenadna/model_code/`.

## Pretraining

Each model's `pretraining/configs/` folder has one JSON file for every pretraining run in the
paper, named `pretrain_<setting>_<variant>.json` with the variant name exactly as in the
paper. Every run uses 12 nodes with 2 A100 GPUs each, 24 GPUs in total, as in the paper. For
example, SP_20% on DB2-MSG is submitted from `dnabert2/pretraining/` with:

```
mkdir -p logs
sbatch -J pretrain_DB2-MSG_SP_20% pretrain_dnabert2.sbatch run_mlm.py configs/pretrain_DB2-MSG_SP_20%.json
```

The first argument is the script that runs the variant and the second is its config:

| Model | Script | Variants |
|---|---|---|
| DNABERT-2 | `run_mlm.py` | Unpruned, SP_*, SP_20%_window_*, RP_20%, LP_0% |
| | `run_mlm_dsi_targeted.py` | DSI_* |
| | `run_mlm_el2n.py` | EL2N_* |
| | `run_mlm_infobatch.py` | Info_* |
| | `run_mlm_bpas.py` | BPAS_* |
| NT-500M | `train_nt.py` | Unpruned, SP_* |
| | `train_nt_bpas.py` | BPAS_* |
| Hyena and Hyena-16k | `run_clm.py` | Unpruned, SP_*, RP_20% |
| | `run_clm_perplexity.py` | Perp_* |
| | `run_clm_bpas.py` | BPAS_* |

Unpruned is the baseline without pruning. SP_20%_window_2 and SP_20%_window_3 average each
sequence's accuracy over its last two or three epochs before deciding. RP_20% trains on a
random part of the corpus, sized so that its DSI matches that of SP_20% (random pruning).
LP_0% does the opposite of SP and removes hard sequences: at every epoch boundary it removes
the sequences with 0% accuracy.
The DSI_* runs choose how many sequences to remove so that the run follows a target DSI
schedule, and accuracy decides which ones. EL2N_* and Perp_* keep the SP framework but use
EL2N or perplexity instead of accuracy. Info_* and BPAS_* are the two dynamic pruning methods
we compare with, InfoBatch and batch pruning by activation stability (B-PAS). The settings of
every variant are in Appendix F of the paper and in its config file.

A pruned run keeps the baseline's number of epochs (`graceful_exit_epoch` in the configs) and
its learning-rate schedule, and it stops earlier if every sequence has been removed. The final
model is saved in `<output_dir>/final_model` for DNABERT-2 and Hyena, and in
`<output_dir>/epoch_99_checkpoint` for NT-500M. An NT-500M B-PAS run that removes every batch
before epoch 99 stops there, and its final model is the last `checkpoint-*` folder.

Each sequence has a `line_id`, its position among the non-empty lines of `train.txt`. It does
not depend on the pruning settings, so the same `line_id` is the same sequence in every run on
a corpus, and runs can be compared sequence by sequence. At the end of every epoch, the scripts
write what they measured and what they removed into a tracking folder inside `output_dir`:
`mlm_tracking/` for DNABERT-2 and NT-500M, `clm_tracking/` for Hyena, and `bpas_tracking/` for
the B-PAS runs. The analysis scripts read these folders.

DNABERT-2 pretraining reads the model config of `zhihan1996/DNABERT-2-117M` from the Hugging
Face Hub when it starts, so the nodes need internet access for that. If a job stops early,
submit the same command again, and training resumes from the latest checkpoint in
`output_dir`.

## Fine-tuning

| Model | Benchmark | Script |
|---|---|---|
| DNABERT-2 | GUE, NT-18 | `dnabert2/finetuning/finetune_gue.sbatch`, `finetune_nt18.sbatch` |
| NT-500M | GUE, NT-18 | `nucleotide_transformer/finetuning/finetune_gue.sbatch`, `finetune_nt18.sbatch` |
| Hyena | GB | `hyenadna/finetuning/finetune_gb.sbatch` |

Each script takes the folder of a pretrained model and a name for the run, and fine-tunes every
task of the benchmark with the settings in Tables 10 and 11 of the paper, one GPU per SLURM
array task. From `dnabert2/finetuning/`:

```
mkdir -p logs
sbatch -J finetune_gue_DB2-MSG_SP_20% finetune_gue.sbatch \
    /path/to/outputs/pretrain_DB2-MSG_SP_20%/final_model DB2-MSG_SP_20%
```

When all tasks have finished, `master_summary.txt` and `master_summary.csv` in the run's output
folder list the test scores of every task and the score of the benchmark as the paper reports
it: the mean of the group averages for GUE and NT-18, and the mean of the seven tasks for GB.

The Full pruned baseline of the paper is fine-tuning with no pretraining at all, which is the
limit of pruning everything. Add `random_init` as a third argument, and the model is built
with random weights. The model folder you pass is then used only for its config and
tokenizer, so give one from the same setting:

```
sbatch -J finetune_gue_DB2-MSG_Full_pruned finetune_gue.sbatch \
    /path/to/outputs/pretrain_DB2-MSG_Unpruned/final_model DB2-MSG_Full_pruned random_init
```

Every pretraining and fine-tuning run uses seed 42. The seed-variance runs in Appendix G add
seeds 43 and 44: Table 17 fine-tunes the same pretrained model with seeds 42, 43 and 44, and
Table 18 pretrains SP_20% with seeds 42, 43 and 44. To pretrain with another seed, copy the
config, change `seed` and also change `output_dir`, because a run that finds checkpoints in its
`output_dir` resumes from them instead of starting fresh. To fine-tune with another seed, set
`--seed` in the training command of the fine-tuning `.sbatch` file and give the run its own
name, such as `DB2-MSG_SP_20%_seed43`, so its results go to a separate folder.

## Frozen probing

The Hyena-16k models are evaluated on BEND gene finding and DART-Eval Task 4 with each
benchmark's own code. `hyenadna/frozen_probing/` has the files that let the two benchmarks
load our models, and its [README](hyenadna/frozen_probing/README.md) explains the setup, the
data download and how to run them.

## Analysis

`analysis/compute_dsi.py` computes the DSI of one pretraining run from its tracking folder,
for every method in the paper. It reports the number of batches trained in each epoch, their
sum, the denominator (scheduled epochs times the batches in one full epoch) and the DSI:

```
python analysis/compute_dsi.py --tracking_dir /path/to/outputs/pretrain_DB2-MSG_SP_20% \
    --scheduled_epochs 47 --batch_size 3072 \
    --output DB2-MSG_SP_20%_dsi.txt --per_epoch_csv DB2-MSG_SP_20%_dsi_per_epoch.csv
```

`--scheduled_epochs` is the number of epochs of the unpruned baseline, and `--batch_size` is the
global batch size. B-PAS runs do not need `--batch_size`.

| Setting | Scheduled epochs | Global batch size |
|---|---|---|
| DB2-MSG | 47 | 3072 |
| DB2-HRG | 32 | 3072 |
| NT-HRG | 99 | 144 |
| HY-HRG | 28 | 1536 |
| HY16k-HRG | 2 | 96 |

`--per_epoch_csv` also writes the DSI reached after each epoch.

`analysis/plot_accuracy_figures.py` draws Figures 1 and 2 from the tracking folders of the
unpruned DB2-MSG and HY-HRG runs, and `analysis/plot_pruning_dynamics.py` draws Figure 6 from
an unpruned and a pruned run of each. The commands are at the top of each script.

## Pretrained models

Coming soon on Hugging Face.

## Citation

Coming soon.

## Acknowledgments

This code builds on the code and data of
[DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2) and
[MosaicBERT](https://github.com/mosaicml/examples),
[Nucleotide Transformer](https://github.com/instadeepai/nucleotide-transformer),
[HyenaDNA](https://github.com/HazyResearch/hyena-dna),
[InfoBatch](https://github.com/NUS-HPC-AI-Lab/InfoBatch),
B-PAS ([Alam et al., 2026](https://openreview.net/forum?id=TUADW7db5n)),
[BEND](https://github.com/frederikkemarin/BEND),
[DART-Eval](https://github.com/kundajelab/DART-Eval) and
[Genomic Benchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks).
We thank their authors for releasing them.

## License

This repository is released under the [Apache License 2.0](LICENSE). Files that were copied
or adapted from the projects above keep the license of the project they came from.
