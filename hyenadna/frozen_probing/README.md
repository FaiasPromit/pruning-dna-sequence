# Frozen probing: BEND gene finding and DART-Eval Task 4

In the paper, the Hyena-16k models (HY16k-HRG) are evaluated on BEND gene finding (B-GF)
and DART-Eval Task 4 (D-T4) by frozen probing. The pretrained model is only used to turn
each sequence into embeddings, and its weights never change; only the benchmark's own
small model on top of the embeddings (the probe) is trained. Both benchmarks are run with
their own code and their own settings. The files here add one thing to each benchmark:
a loader for our models, because the loaders that come with BEND and DART-Eval only work
for the official HyenaDNA checkpoints.

Each benchmark lives in its own clone and has its own conda environment. Our files for each
benchmark are in `bend/files_for_bend_repo/` and `dart_eval/files_for_dart_repo/`, and the
sections below show how to copy them into the benchmark's clone.

## BEND gene finding (B-GF)

### One-time setup

Get BEND and build its environment:

```
git clone https://github.com/frederikkemarin/BEND.git /path/to/BEND
cd /path/to/BEND
conda create --prefix /path/to/envs/bend python=3.10 -y
conda activate /path/to/envs/bend
pip install -r requirements.txt
pip install -e .
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers lxml pysam h5py "numpy<2" "setuptools<81"
```

The two version limits are needed: some of BEND's packages fail with NumPy 2, and
setuptools 81 and later no longer include pkg_resources, which pyfaidx and wandb still use.

Copy our files into BEND, then download the data from the BEND folder. `files_for_bend_repo/`
has the same layout as the BEND repository, so one `cp` command puts the files in place. It
adds `bend/utils/hyena16k_hrg_embedder.py` and replaces `scripts/download_bend.py` with a
version that works around a link-parsing problem in the original.

```
cp -r /path/to/pruning-dna-sequence/hyenadna/frozen_probing/bend/files_for_bend_repo/. /path/to/BEND/
cd /path/to/BEND
python scripts/download_bend.py
```

### Adding a model

BEND knows a model by a name in two of its config files. For each pretrained model, add a
block to `conf/embedding/embed.yaml`:

```
HY16k-HRG_SP_20:
  _target_ : bend.utils.hyena16k_hrg_embedder.Hyena16kHRGEmbedder
  model_path : /path/to/outputs/pretrain_HY16k-HRG_SP_20%/final_model
  model_code_dir : /path/to/pruning-dna-sequence/hyenadna/model_code
  upsample_embeddings : false
```

and a line with its embedding size (256 for every Hyena model) to
`conf/datadims/embedding_dims.yaml`:

```
HY16k-HRG_SP_20 : 256
```

The BEND names leave out the % of the paper's variant names, to be safe, because BEND
reads these names through its Hydra config files and uses them in file paths.

### Running

From `hyenadna/frozen_probing/bend/`:

```
mkdir -p logs
sbatch -J bend_gene_finding_HY16k-HRG_SP_20 bend_gene_finding.sbatch HY16k-HRG_SP_20
```

The test MCC is printed at the end of the log and saved in
`BEND/downstream_tasks/gene_finding/<name>/best_model_metrics.csv`.

## DART-Eval Task 4 (D-T4)

### One-time setup

Get DART-Eval and build its environment. This follows DART-Eval's `environment.yml`
without mamba-ssm, which is only needed for the Caduceus model and has to compile CUDA
code, and with synapseclient, which is needed to download the data:

```
git clone https://github.com/kundajelab/DART-Eval.git /path/to/DART-Eval
cd /path/to/DART-Eval
conda create --prefix /path/to/envs/dart python=3.10 -y
conda activate /path/to/envs/dart
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.3 scipy==1.12.0 polars==0.20.17 pandas==2.2.0 transformers==4.37.1 \
    tqdm==4.66.1 h5py==3.10.0 scikit-learn==1.4.2 pyfaidx==0.7.2.1 pyBigWig==0.3.22 \
    ncls==0.0.68 synapseclient einops "setuptools<81"
pip install -e .
```

Copy our four files into DART-Eval. `hyena16k_hrg_embeddings.py` is the model loader. The
other three are DART-Eval's own HyenaDNA scripts for Task 4, the ones that extract the
embeddings, train the probe and evaluate it, with the model read from environment variables,
so the probe and its settings are unchanged. In DART-Eval each of the three is named
`hyena16k_hrg.py` and goes in its own folder, so the copy commands rename them. We keep the
four files in one folder here because DART-Eval's folders are nested so deeply that copying
its layout makes the paths too long for Windows.

```
FILES=/path/to/pruning-dna-sequence/hyenadna/frozen_probing/dart_eval/files_for_dart_repo
DART_SRC=/path/to/DART-Eval/src/dnalm_bench/task_2_5_single
cp $FILES/hyena16k_hrg_embeddings.py $DART_SRC/hyena16k_hrg_embeddings.py
cp $FILES/task4_extract_embeddings.py $DART_SRC/experiments/task_4_chromatin_activity/extract_embeddings/hyena16k_hrg.py
cp $FILES/task4_train.py $DART_SRC/experiments/task_4_chromatin_activity/train/hyena16k_hrg.py
cp $FILES/task4_eval_probing.py $DART_SRC/experiments/task_4_chromatin_activity/eval_probing/hyena16k_hrg.py
```

### Data

The data is on Synapse, so you need a Synapse account and a personal access token
(Synapse web site, Account Settings, Personal Access Tokens). `dart_download_task4.sbatch`
downloads only the genome and the Task 4 labels, a few GB instead of the full Task 4
folder. From `hyenadna/frozen_probing/dart_eval/`:

```
export SYNAPSE_AUTH_TOKEN=<your token>
mkdir -p logs
sbatch --export=ALL dart_download_task4.sbatch
```

### Running

From `hyenadna/frozen_probing/dart_eval/`, with the model folder and a name for the run
(K562 is the default cell line, as in the paper):

```
sbatch -J dart_task4_HY16k-HRG_SP_20% dart_task4.sbatch \
    /path/to/outputs/pretrain_HY16k-HRG_SP_20%/final_model HY16k-HRG_SP_20%
```

The metrics are printed at the end of the log and saved in
`DART-Eval/work/task_4_chromatin_activity/supervised_model_outputs/probed/<run name>/K562/eval_test.json`.
The D-T4 score in the paper is `test_spearman_all`, the Spearman correlation over the
peak and non-peak test regions together.
