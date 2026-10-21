#!/usr/bin/env python3
"""
Data Savings Index (DSI) of one pretraining run, computed from the files that the
pretraining scripts write to the run's tracking folder at the end of every epoch.

    DSI = 1 - (n_1 + n_2 + ... + n_k) / (I * n_0)

    n_i  batches trained in epoch i (k is the last epoch that was run)
    n_0  batches in one epoch over the whole corpus, that is, one unpruned epoch
    I    scheduled number of epochs, the epoch count of the unpruned baseline:
         47 for DB2-MSG, 32 for DB2-HRG, 99 for NT-HRG, 28 for HY-HRG, 2 for HY16k-HRG

This is the DSI of the paper (Section 3) counted in batches, where a batch is one
global batch of per_device_train_batch_size x number of GPUs sequences. DSI = 50%
means that half of the sequence processing of the unpruned baseline was avoided.

The script reads three kinds of tracking folder:

  B-PAS (epoch_*_bpas_stats.txt; run_mlm_bpas.py, train_nt_bpas.py, run_clm_bpas.py)
      n_i  batches still active at the start of epoch i: the "active" count in the
           file of epoch i (taken after that epoch's pruning) plus "newly_pruned"
      n_0  number of fixed global batches, the N in "total_pruned=X/N"

  InfoBatch (epoch_*_infobatch_table.txt; run_mlm_infobatch.py)
      n_i  "sequences trained" divided by the batch size, checked against the
           "optimizer steps this epoch" written in the same table
      n_0  "steps if full dataset"

  Every other method: SP, RP, DSI schedules, EL2N, perplexity (epoch_*_pruning_table.txt,
  epoch_*_accuracy_dist.txt, epoch_*_perplexity_dist.txt or epoch_*_el2n_dist.txt;
  run_mlm.py, run_mlm_dsi_targeted.py, run_mlm_el2n.py, train_nt.py, run_clm.py,
  run_clm_perplexity.py)
      n_i  "Processed/tracked this epoch" divided by the batch size
      n_0  corpus size divided by the batch size, where the corpus size is
           "Total pruned so far" + "Active sequences next epoch" in a pruning table.
           This is also right for RP_20%, whose first epoch is already reduced.

Usage (the folder can be the tracking folder or the run's output_dir):
    python compute_dsi.py --tracking_dir /path/to/outputs/pretrain_DB2-MSG_SP_20% \\
        --scheduled_epochs 47 --batch_size 3072 \\
        --output DB2-MSG_SP_20%_dsi.txt --per_epoch_csv DB2-MSG_SP_20%_dsi_per_epoch.csv

--batch_size is the global batch size (per_device_train_batch_size x number of GPUs):
3072 for DB2-MSG and DB2-HRG, 144 for NT-HRG, 1536 for HY-HRG and 96 for HY16k-HRG.
B-PAS runs do not need it. The per-epoch CSV gives the DSI reached after each epoch,
1 - (n_1 + ... + n_j) / (j * n_0) after epoch j, which at the scheduled last epoch equals
the DSI of the run.
"""

import argparse
import csv
import glob
import os
import re
import sys

TRACKING_SUBFOLDERS = ("mlm_tracking", "clm_tracking", "bpas_tracking")
SEQUENCE_FILE_TYPES = ("pruning_table", "accuracy_dist", "perplexity_dist", "el2n_dist")


def find_tracking_dir(path):
    """Return the folder that holds the epoch files: path itself or one of its tracking subfolders."""
    if glob.glob(os.path.join(path, "epoch_*_*.txt")):
        return path
    for sub in TRACKING_SUBFOLDERS:
        d = os.path.join(path, sub)
        if glob.glob(os.path.join(d, "epoch_*_*.txt")):
            return d
    sys.exit(f"ERROR: no epoch_*.txt tracking files in {path} or in its "
             f"{', '.join(TRACKING_SUBFOLDERS)} subfolders")


def epoch_files(tracking_dir, file_type):
    """{epoch number: path} for epoch_<n>_<file_type>.txt (zero-padded or not)."""
    out = {}
    for f in glob.glob(os.path.join(tracking_dir, f"epoch_*_{file_type}.txt")):
        m = re.match(rf"epoch_(\d+)_{file_type}\.txt$", os.path.basename(f))
        if m:
            out[int(m.group(1))] = f
    return out


def detect_method(tracking_dir):
    if epoch_files(tracking_dir, "bpas_stats"):
        return "bpas"
    if epoch_files(tracking_dir, "infobatch_table"):
        return "infobatch"
    if any(epoch_files(tracking_dir, t) for t in SEQUENCE_FILE_TYPES):
        return "sequence"
    sys.exit(f"ERROR: no B-PAS, InfoBatch or pruning files in {tracking_dir}")


def first_number_after(text, label):
    """The integer that follows 'label:' on its line, commas removed, or None."""
    m = re.search(rf"^\s*{re.escape(label)}\s*:?\s*([\d,]+)", text, re.MULTILINE)
    return int(m.group(1).replace(",", "")) if m else None


# B-PAS

RE_BPAS_NEWLY = re.compile(r"newly_pruned=(\d+)")
RE_BPAS_TOTAL = re.compile(r"total_pruned=(\d+)\s*/\s*(\d+)")
RE_BPAS_ACTIVE = re.compile(r"active=(\d+)\s+batches\s*\(([\d,]+)\s+seqs\)")
RE_BPAS_GPU = re.compile(r"GPU-hours=([\d.]+)")


def read_bpas(tracking_dir, batch_size, warnings):
    epochs = []
    n0_seen = set()
    gpu_hours = None
    for ep, path in sorted(epoch_files(tracking_dir, "bpas_stats").items()):
        header = None
        with open(path) as f:
            for line in f:
                if line.lstrip().startswith("B-PAS Epoch"):
                    header = line.strip()
                    break
        m_new = RE_BPAS_NEWLY.search(header or "")
        m_tot = RE_BPAS_TOTAL.search(header or "")
        m_act = RE_BPAS_ACTIVE.search(header or "")
        if not (m_new and m_tot and m_act):
            warnings.append(f"Epoch {ep}: could not read the B-PAS header of {path}")
            continue
        active_after = int(m_act.group(1))
        seqs_after = int(m_act.group(2).replace(",", ""))
        n_i = active_after + int(m_new.group(1))
        n0_seen.add(int(m_tot.group(2)))
        if active_after > 0 and batch_size and seqs_after != active_after * batch_size:
            warnings.append(f"Epoch {ep}: --batch_size {batch_size} differs from the global batch "
                            f"in the file ({seqs_after:,} seqs / {active_after:,} batches)")
        m_gpu = RE_BPAS_GPU.search(header)
        if m_gpu:
            gpu_hours = float(m_gpu.group(1))
        epochs.append((ep, n_i, ""))
    if len(n0_seen) > 1:
        warnings.append(f"Different total batch counts across epochs: {sorted(n0_seen)}")
    n0 = min(n0_seen) if n0_seen else None
    return epochs, n0, "number of fixed global batches (total_pruned=X/N)", gpu_hours


# InfoBatch

INFOBATCH_FIELDS = {
    "steps": "optimizer steps this epoch",
    "steps_full": "steps if full dataset",
    "trained": "sequences trained",
}


def read_infobatch(tracking_dir, batch_size, warnings, errors):
    epochs = []
    n0 = None
    for ep, path in sorted(epoch_files(tracking_dir, "infobatch_table").items()):
        text = open(path).read()
        d = {k: first_number_after(text, label) for k, label in INFOBATCH_FIELDS.items()}
        if any(v is None for v in d.values()):
            warnings.append(f"Epoch {ep}: could not read {path}")
            continue
        n_i, rem = divmod(d["trained"], batch_size)
        if rem:
            warnings.append(f"Epoch {ep}: {d['trained']:,} sequences trained is not a multiple "
                            f"of the batch size {batch_size} (remainder {rem})")
        if n_i != d["steps"]:
            errors.append(f"Epoch {ep}: {n_i:,} batches from sequences / batch size, but the table "
                          f"has {d['steps']:,} optimizer steps; --batch_size is probably wrong "
                          f"(implied {d['trained'] // max(1, d['steps']):,})")
        mode = "anneal" if re.search(r"annealing active this epoch\s*:\s*True", text) else "prune"
        if n0 is None:
            n0 = d["steps_full"]
        elif d["steps_full"] != n0:
            warnings.append(f"Epoch {ep}: steps if full dataset = {d['steps_full']:,}, "
                            f"first epoch had {n0:,}")
        epochs.append((ep, n_i, mode))
    return epochs, n0, "steps if full dataset (InfoBatch table)", None


# Sequence pruning and the other methods

def read_sequence(tracking_dir, batch_size, warnings):
    files = {t: epoch_files(tracking_dir, t) for t in SEQUENCE_FILE_TYPES}
    all_epochs = sorted(set().union(*[set(v) for v in files.values()]))
    epochs = []
    corpus_size = None
    for ep in all_epochs:
        processed = None
        for t in SEQUENCE_FILE_TYPES:
            path = files[t].get(ep)
            if not path:
                continue
            text = open(path).read()
            if processed is None:
                processed = first_number_after(text, "Processed/tracked this epoch")
            if corpus_size is None and t == "pruning_table":
                pruned = first_number_after(text, "Total pruned so far")
                active = first_number_after(text, "Active sequences next epoch")
                if pruned is not None and active is not None:
                    corpus_size = pruned + active
        if processed is None:
            warnings.append(f"Epoch {ep}: no 'Processed/tracked this epoch' line found")
            continue
        n_i, rem = divmod(processed, batch_size)
        if rem:
            warnings.append(f"Epoch {ep}: {processed:,} sequences is not a multiple of the batch "
                            f"size {batch_size} (remainder {rem})")
        epochs.append((ep, n_i, ""))
    if corpus_size is None:
        return epochs, None, None, None
    return (epochs, corpus_size // batch_size,
            f"corpus size {corpus_size:,} / batch size {batch_size}", None)


def main():
    ap = argparse.ArgumentParser(description="Compute the DSI of one pretraining run.")
    ap.add_argument("--tracking_dir", required=True,
                    help="The run's tracking folder (mlm_tracking, clm_tracking or bpas_tracking) "
                         "or its output_dir")
    ap.add_argument("--scheduled_epochs", type=int, required=True,
                    help="I, the epoch count of the unpruned baseline")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="Global batch size (per_device_train_batch_size x number of GPUs); "
                         "not needed for B-PAS")
    ap.add_argument("--batches_per_epoch", type=int, default=None,
                    help="Use this n_0 instead of reading it from the files")
    ap.add_argument("--start_epoch", type=int, default=None, help="First epoch to count (default: all)")
    ap.add_argument("--end_epoch", type=int, default=None, help="Last epoch to count (default: all)")
    ap.add_argument("--output", default=None, help="Write the summary to this text file")
    ap.add_argument("--per_epoch_csv", default=None, help="Write the DSI after each epoch to this CSV")
    args = ap.parse_args()

    tracking_dir = find_tracking_dir(args.tracking_dir)
    method = detect_method(tracking_dir)
    if method != "bpas" and not args.batch_size:
        sys.exit("ERROR: --batch_size is needed for this run (it is not a B-PAS run)")

    warnings, errors = [], []
    if method == "bpas":
        epochs, n0, n0_source, gpu_hours = read_bpas(tracking_dir, args.batch_size, warnings)
    elif method == "infobatch":
        epochs, n0, n0_source, gpu_hours = read_infobatch(tracking_dir, args.batch_size, warnings, errors)
    else:
        epochs, n0, n0_source, gpu_hours = read_sequence(tracking_dir, args.batch_size, warnings)

    if args.batches_per_epoch:
        n0, n0_source = args.batches_per_epoch, "--batches_per_epoch"
    if args.start_epoch is not None:
        epochs = [e for e in epochs if e[0] >= args.start_epoch]
    if args.end_epoch is not None:
        epochs = [e for e in epochs if e[0] <= args.end_epoch]
    if not epochs:
        sys.exit(f"ERROR: no epochs could be read from {tracking_dir}")
    if not n0:
        sys.exit("ERROR: could not find n_0 (batches in one full epoch); pass --batches_per_epoch")

    numbers = [e[0] for e in epochs]
    gaps = sorted(set(range(numbers[0], numbers[-1] + 1)) - set(numbers))
    if gaps:
        warnings.append(f"Epoch files missing for {gaps}: their batches are not counted")
    if len(epochs) > args.scheduled_epochs:
        warnings.append(f"{len(epochs)} epochs were found but --scheduled_epochs is "
                        f"{args.scheduled_epochs}; check the epoch range")

    numerator = sum(e[1] for e in epochs)
    denominator = args.scheduled_epochs * n0
    dsi = 1.0 - numerator / denominator

    # DSI reached after each epoch
    rows, cum = [], 0
    for j, (ep, n_i, mode) in enumerate(epochs, start=1):
        cum += n_i
        rows.append((ep, n_i, cum, j * n0, 1.0 - cum / (j * n0), mode))

    method_name = {"bpas": "B-PAS", "infobatch": "InfoBatch",
                   "sequence": "sequence pruning (or another per-sequence method)"}[method]
    lines = [
        "=" * 78,
        "DSI summary",
        "=" * 78,
        f"Tracking folder            : {tracking_dir}",
        f"Method                     : {method_name}",
        f"Epochs counted             : {len(epochs)} (epoch {numbers[0]} to {numbers[-1]})",
        f"Scheduled epochs I         : {args.scheduled_epochs}",
        f"Batches per full epoch n_0 : {n0:,} ({n0_source})",
    ]
    if args.batch_size:
        lines.append(f"Global batch size          : {args.batch_size:,}")
    if gpu_hours is not None:
        lines.append(f"GPU-hours (last epoch file): {gpu_hours:.2f}")
    if errors:
        lines += ["", "ERRORS:"] + [f"  {e}" for e in errors]
    if warnings:
        lines += ["", "WARNINGS:"] + [f"  {w}" for w in warnings]
    lines += [
        "-" * 78,
        f"Numerator   (sum of n_i)   : {numerator:,}",
        f"Denominator (I x n_0)      : {denominator:,}",
        f"DSI = 1 - num / denom      : {dsi:.6f} ({100 * dsi:.2f}%)",
        "-" * 78,
        "",
        "Per epoch:",
        f"  {'epoch':>6}  {'n_i':>12}  {'sum of n_i':>14}  {'j x n_0':>14}  {'DSI so far':>10}"
        + ("  mode" if method == "infobatch" else ""),
        "  " + "-" * 64,
    ]
    for ep, n_i, c, d, v, mode in rows:
        lines.append(f"  {ep:>6}  {n_i:>12,}  {c:>14,}  {d:>14,}  {100 * v:>9.2f}%"
                     + (f"  {mode}" if method == "infobatch" else ""))
    text = "\n".join(lines) + "\n"
    print(text)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"Summary written to {args.output}")
    if args.per_epoch_csv:
        with open(args.per_epoch_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "n_i", "sum_n_i", "denom", "DSI"])
            for ep, n_i, c, d, v, _ in rows:
                w.writerow([ep, n_i, c, d, f"{v:.6f}"])
        print(f"Per-epoch DSI written to {args.per_epoch_csv}")
    if errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
