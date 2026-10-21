#!/usr/bin/env python3
"""
Download the 18 Nucleotide Transformer downstream tasks (NT-18) and write each task as
train.csv, dev.csv and test.csv for train.py.

The tasks come from the Hugging Face dataset
InstaDeepAI/nucleotide_transformer_downstream_tasks_revised, which has a train and a
test split per task. The parquet files are downloaded directly, and 10% of each train
split is set aside as the dev split (stratified by label, seed 42). The CSV files have
the two columns sequence,label, the same format as GUE.

Run it once on a machine with internet access (for example a login node):
    python prepare_nt18.py --out /path/to/NT18

Output:
    <out>/
        H2AFZ/
            _raw/train.parquet   _raw/test.parquet   (downloaded files)
            train.csv  dev.csv  test.csv
        H3K27ac/
        ...                                          (18 task folders)

--cleanup_raw deletes the _raw/ folders after the CSV files are written.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import train_test_split


# Task folder names in the Hugging Face dataset repository
TASK_NAMES = [
    "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
    "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3", "H4K20me1",
    "enhancers", "enhancers_types",
    "promoter_all", "promoter_no_tata", "promoter_tata",
    "splice_sites_acceptors", "splice_sites_all", "splice_sites_donors",
]

REPO_BASE = (
    "https://huggingface.co/datasets/"
    "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/"
    "resolve/main"
)


def download(url: str, dest: Path) -> None:
    """Download url to dest, through a temporary .partial file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    with urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    tmp.rename(dest)  # an interrupted download never leaves a broken file behind


def prepare_task(task: str, out_root: Path, dev_frac: float, seed: int) -> None:
    task_dir = out_root / task
    raw_dir  = task_dir / "_raw"
    task_dir.mkdir(parents=True, exist_ok=True)

    train_pq = raw_dir / "train.parquet"
    test_pq  = raw_dir / "test.parquet"

    if not train_pq.exists():
        print(f"  [dl] {task}/train.parquet")
        download(f"{REPO_BASE}/{task}/train.parquet", train_pq)
    if not test_pq.exists():
        print(f"  [dl] {task}/test.parquet")
        download(f"{REPO_BASE}/{task}/test.parquet", test_pq)

    # The parquet files have the columns sequence, name, label and task; train.py needs two
    train_df = pd.read_parquet(train_pq)[["sequence", "label"]]
    test_df  = pd.read_parquet(test_pq)[["sequence", "label"]]

    try:
        tr_df, dv_df = train_test_split(
            train_df, test_size=dev_frac,
            stratify=train_df["label"], random_state=seed,
        )
    except ValueError:
        # A class is too rare to stratify; this does not happen for NT-18, but the split
        # still works without stratification
        tr_df, dv_df = train_test_split(
            train_df, test_size=dev_frac, random_state=seed,
        )

    tr_df.to_csv(task_dir / "train.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    dv_df.to_csv(task_dir / "dev.csv",   index=False, quoting=csv.QUOTE_MINIMAL)
    test_df.to_csv(task_dir / "test.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    labels = sorted(set(train_df["label"].tolist()))
    print(f"  [ok] {task:<24}  train={len(tr_df):>6}  dev={len(dv_df):>5}  "
          f"test={len(test_df):>5}  labels={labels}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="Output folder, for example /path/to/NT18")
    ap.add_argument("--dev_frac", type=float, default=0.1,
                    help="Fraction of train to carve out as dev (default 0.1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cleanup_raw", action="store_true",
                    help="Delete per-task _raw/ parquets after CSVs are written.")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[prep] output root: {out_root}")
    print(f"[prep] downloading the 18 tasks from Hugging Face")
    print(f"[prep] dev split: stratified {args.dev_frac:.0%} of train, seed={args.seed}")
    print()

    for task in TASK_NAMES:
        prepare_task(task, out_root, args.dev_frac, args.seed)

    if args.cleanup_raw:
        print()
        print("[prep] cleaning up _raw/ staging dirs")
        for task in TASK_NAMES:
            shutil.rmtree(out_root / task / "_raw", ignore_errors=True)

    print()
    print("[prep] done.")
    print(f"[prep] verify with: head -3 {out_root}/promoter_all/train.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
