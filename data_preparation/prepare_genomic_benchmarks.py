#!/usr/bin/env python
"""
Download the 7 Genomic Benchmarks (GB) datasets used in the paper and write each one as
train.csv, dev.csv and test.csv for hyenadna/finetuning, in the same format as GUE and
NT-18 (header row, columns sequence,label).

Genomic Benchmarks has only a train and a test split, so 10% of each train split is set
aside as the dev split (stratified by label, seed 42) and used to pick the best
checkpoint. The test split is used as it is.

    dataset                            sequences   classes   median length
    demo_coding_vs_intergenomic_seqs      100,000         2             200
    demo_human_or_worm                    100,000         2             200
    human_enhancers_cohn                   27,791         2             500
    human_enhancers_ensembl               154,842         2             269
    human_ensembl_regulatory              289,061         3             401
    human_nontata_promoters                36,131         2             251
    human_ocr_ensembl                     174,756         2             315

Requires the genomic-benchmarks package (pip install genomic-benchmarks). Run it once
on a machine with internet access (for example a login node):
    python prepare_genomic_benchmarks.py --output_dir /path/to/Genomic_Benchmarks

Output:
    <output_dir>/<dataset>/train.csv, dev.csv, test.csv, label_map.json
"""

import argparse
import csv
import json
import os
from pathlib import Path

DATASETS = [
    "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl",
]


def locate_dataset_root(name, returned_path):
    """Find the folder that contains the train/ and test/ folders of a dataset."""
    candidates = []
    if returned_path is not None:
        rp = Path(returned_path)
        candidates += [rp, rp / name]
    candidates += [
        Path.home() / ".genomic_benchmarks" / name,
        Path.home() / ".genomic_benchmarks" / "datasets" / name,
    ]
    for c in candidates:
        if (c / "train").is_dir() and (c / "test").is_dir():
            return c
    raise FileNotFoundError(
        f"Could not locate train/ and test/ for '{name}'. Looked in: "
        + ", ".join(str(c) for c in candidates))


def read_split(root: Path, split: str):
    """Read <root>/<split>/<class>/*.txt and return (sequences, labels, class name -> label)."""
    split_dir = root / split
    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
    sequences, labels = [], []
    for d in class_dirs:
        idx = class_to_idx[d.name]
        # Sorted, so the train/dev split is the same on every machine
        for txt in sorted(d.glob("*.txt")):
            seq = txt.read_text().strip().upper().replace("\n", "")
            if seq:
                sequences.append(seq)
                labels.append(idx)
    return sequences, labels, class_to_idx


def write_csv(path, sequences, labels):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "label"])
        for s, y in zip(sequences, labels):
            w.writerow([s, y])


def prepare_one(name, out_dir, dev_frac, seed):
    from genomic_benchmarks.loc2seq import download_dataset
    from sklearn.model_selection import train_test_split

    print(f"\n=== {name} ===")
    try:
        returned = download_dataset(name, version=0)
    except TypeError:
        returned = download_dataset(name)  # some package versions take no version argument
    root = locate_dataset_root(name, returned)
    print(f"  dataset folder: {root}")

    train_seqs, train_labels, c2i = read_split(root, "train")
    test_seqs, test_labels, c2i_test = read_split(root, "test")
    assert c2i == c2i_test, f"train and test have different classes for {name}"

    # Stratified split of train into train and dev
    tr_s, dv_s, tr_y, dv_y = train_test_split(
        train_seqs, train_labels, test_size=dev_frac,
        random_state=seed, stratify=train_labels)

    task_out = os.path.join(out_dir, name)
    os.makedirs(task_out, exist_ok=True)
    write_csv(os.path.join(task_out, "train.csv"), tr_s, tr_y)
    write_csv(os.path.join(task_out, "dev.csv"), dv_s, dv_y)
    write_csv(os.path.join(task_out, "test.csv"), test_seqs, test_labels)
    with open(os.path.join(task_out, "label_map.json"), "w") as f:
        json.dump(c2i, f, indent=2)

    lens = [len(s) for s in (tr_s + dv_s + test_seqs)]
    print(f"  classes={len(c2i)} | train={len(tr_s)} dev={len(dv_s)} test={len(test_seqs)} "
          f"| length min/median/max={min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dev_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", nargs="*", default=DATASETS,
                    help="Datasets to prepare (default: the 7 used in the paper)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for name in args.datasets:
        prepare_one(name, args.output_dir, args.dev_frac, args.seed)
    print(f"\nAll done: {args.output_dir}/<dataset>/{{train,dev,test}}.csv")


if __name__ == "__main__":
    main()
