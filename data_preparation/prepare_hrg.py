#!/usr/bin/env python
"""
Download the human reference genome (GRCh38.p14 from NCBI) and write the HRG pretraining
corpora used in the paper, one sequence per line:

  DB2-HRG/  1,000 bp sequences without overlap, for DNABERT-2. Sequences that contain any
            N are dropped, as in the DNABERT-2 paper.
  NT-HRG/   6,100 bp windows that start every 6,000 bp, for NT-500M, following the
            Nucleotide Transformer paper. Neighboring windows share 100 bp, that is 50 bp
            of overlap on each side of each 6,000 bp core. N is kept.
  HY-HRG/   1,024 bp windows without overlap, for Hyena. N is kept. Chromosomes 14 and X
            are held out in test.txt, as in the HyenaDNA paper; test.txt is not used in
            pretraining.
  HY16k-HRG/ 16,384 bp windows without overlap, for Hyena-16k. N is kept.

All use chromosomes 1-22, X and Y of the primary assembly, with every base other than
A, C, G and T written as N, and drop a trailing piece shorter than one window.
Chromosome 21 goes to dev.txt and the other chromosomes to train.txt (except the
test.txt chromosomes of HY-HRG).

Usage:
    python prepare_hrg.py --output_dir /path/to/HRG_corpus
    python prepare_hrg.py --output_dir /path/to/HRG_corpus --corpora HY-HRG HY16k-HRG

The genome is downloaded with wget into output_dir and reused if it is already there, so
on a cluster without internet on the compute nodes you can download
GCF_000001405.40_GRCh38.p14_genomic.fna.gz first (saved as GRCh38.p14_genomic.fna.gz).
"""

import argparse
import gzip
import os
import re
import shutil
import subprocess


FASTA_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/"
    "GCF_000001405.40_GRCh38.p14/"
    "GCF_000001405.40_GRCh38.p14_genomic.fna.gz"
)

VALID_CHROMOSOMES = set(str(i) for i in range(1, 23)) | {"X", "Y"}
VAL_CHROMOSOME = "21"
HY_TEST_CHROMOSOMES = {"14", "X"}   # held out in HY-HRG, as in the HyenaDNA paper

CORPORA = ["DB2-HRG", "NT-HRG", "HY-HRG", "HY16k-HRG"]

HEADER_RE = re.compile(
    r"NC_\d+\.\d+ Homo sapiens chromosome (\d+|[XY]), GRCh38\.p14 Primary Assembly"
)


def clean_sequence(seq: str) -> str:
    """Upper-case and replace every base other than A, T, C and G with N."""
    out = []
    for c in seq.upper():
        out.append(c if c in "ATCG" else "N")
    return "".join(out)


def parse_fasta(fasta_path: str):
    """Yield (chromosome, sequence) for chromosomes 1-22, X and Y of the primary assembly."""
    current_chr = None
    parts = []

    def _flush():
        if current_chr and current_chr in VALID_CHROMOSOMES:
            return current_chr, "".join(parts)
        return None, None

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                chrom, seq = _flush()
                if chrom:
                    yield chrom, seq
                m = HEADER_RE.search(line)
                current_chr = m.group(1) if m else None
                parts = []
            else:
                parts.append(line)
    chrom, seq = _flush()
    if chrom:
        yield chrom, seq


def chunk_nt(sequence: str, chunk_len=6000, overlap=100):
    """Split a sequence into windows of chunk_len + overlap bp that start every chunk_len bp.

    With the defaults, the windows are 6,100 bp long and neighboring windows share 100 bp.
    A trailing window shorter than 6,100 bp is dropped.
    """
    half_overlap = overlap // 2
    seq_len = len(sequence)
    num_chunks = (seq_len - 2 * half_overlap) // chunk_len
    if num_chunks <= 0:
        return []

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_len
        end = min(seq_len, (i + 1) * chunk_len + 2 * half_overlap)
        chunk = sequence[start:end]
        if len(chunk) == chunk_len + 2 * half_overlap:
            chunks.append(chunk)
    return chunks


def chunk_db2(sequence: str, chunk_len=1000):
    """Split a sequence into non-overlapping chunks of chunk_len bp (a shorter tail is dropped)."""
    chunks = []
    for i in range(0, len(sequence) - chunk_len + 1, chunk_len):
        chunks.append(sequence[i : i + chunk_len])
    return chunks


def chunk_windows(sequence: str, chunk_len: int):
    """Split a sequence into non-overlapping windows of chunk_len bp (a shorter tail is dropped)."""
    return [sequence[i:i + chunk_len]
            for i in range(0, len(sequence) - chunk_len + 1, chunk_len)]


def download_fasta(output_dir: str) -> str:
    """Download and decompress the genome FASTA, reusing files that are already there."""
    os.makedirs(output_dir, exist_ok=True)
    gz_path = os.path.join(output_dir, "GRCh38.p14_genomic.fna.gz")
    fasta_path = os.path.join(output_dir, "GRCh38.p14_genomic.fna")

    if os.path.exists(fasta_path):
        print(f"FASTA already exists, reusing: {fasta_path}")
        return fasta_path

    if not os.path.exists(gz_path):
        print("Downloading the genome from NCBI...")
        subprocess.run(["wget", "-O", gz_path, FASTA_URL], check=True)
    else:
        print(f"Compressed FASTA already present, reusing: {gz_path}")

    print("Decompressing...")
    with gzip.open(gz_path, "rt") as fin, open(fasta_path, "w") as fout:
        shutil.copyfileobj(fin, fout)
    print(f"FASTA ready: {fasta_path}")
    return fasta_path


def preprocess_db2(fasta_path: str, out_dir: str):
    """DB2-HRG: 1,000 bp chunks without overlap; chunks containing any N are dropped."""
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.txt")
    dev_path = os.path.join(out_dir, "dev.txt")

    train_count, dev_count = 0, 0
    train_skipped, dev_skipped = 0, 0
    with open(train_path, "w") as ftrain, open(dev_path, "w") as fdev:
        for chrom, raw_seq in parse_fasta(fasta_path):
            seq = clean_sequence(raw_seq)
            chunks = chunk_db2(seq, chunk_len=1000)

            kept, skipped = 0, 0
            for chunk in chunks:
                if "N" in chunk:
                    skipped += 1
                    continue
                if chrom == VAL_CHROMOSOME:
                    fdev.write(chunk + "\n")
                    dev_count += 1
                else:
                    ftrain.write(chunk + "\n")
                    train_count += 1
                kept += 1

            if chrom == VAL_CHROMOSOME:
                dev_skipped += skipped
            else:
                train_skipped += skipped

            print(
                f"  chr{chrom}: {len(seq):,} bp -> {len(chunks):,} chunks "
                f"(kept {kept:,}, dropped {skipped:,} containing N)"
            )

    print("DB2-HRG done:")
    print(f"  train.txt: {train_count:,} sequences "
          f"({train_skipped:,} dropped because of N) -> {train_path}")
    print(f"  dev.txt:   {dev_count:,} sequences "
          f"({dev_skipped:,} dropped because of N) -> {dev_path}")


def preprocess_nt(fasta_path: str, out_dir: str):
    """NT-HRG: 6,100 bp windows starting every 6,000 bp; N is kept."""
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.txt")
    dev_path = os.path.join(out_dir, "dev.txt")

    train_count, dev_count = 0, 0
    with open(train_path, "w") as ftrain, open(dev_path, "w") as fdev:
        for chrom, raw_seq in parse_fasta(fasta_path):
            seq = clean_sequence(raw_seq)
            chunks = chunk_nt(seq, chunk_len=6000, overlap=100)
            print(f"  chr{chrom}: {len(seq):,} bp -> {len(chunks):,} windows")
            for chunk in chunks:
                if chrom == VAL_CHROMOSOME:
                    fdev.write(chunk + "\n")
                    dev_count += 1
                else:
                    ftrain.write(chunk + "\n")
                    train_count += 1

    print("NT-HRG done:")
    print(f"  train.txt: {train_count:,} sequences -> {train_path}")
    print(f"  dev.txt:   {dev_count:,} sequences -> {dev_path}")


def preprocess_hyena(fasta_path: str, out_dir: str):
    """HY-HRG: 1,024 bp windows without overlap; N is kept; chromosomes 14 and X go to test.txt."""
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.txt")
    dev_path = os.path.join(out_dir, "dev.txt")
    test_path = os.path.join(out_dir, "test.txt")

    counts = {"train": 0, "dev": 0, "test": 0}
    with open(train_path, "w") as ftrain, open(dev_path, "w") as fdev, open(test_path, "w") as ftest:
        for chrom, raw_seq in parse_fasta(fasta_path):
            seq = clean_sequence(raw_seq)
            windows = chunk_windows(seq, 1024)

            if chrom in HY_TEST_CHROMOSOMES:
                split, fh = "test", ftest
            elif chrom == VAL_CHROMOSOME:
                split, fh = "dev", fdev
            else:
                split, fh = "train", ftrain

            for w in windows:
                fh.write(w + "\n")
            counts[split] += len(windows)
            print(f"  chr{chrom}: {len(seq):,} bp -> {len(windows):,} windows -> {split}")

    print("HY-HRG done:")
    print(f"  train.txt: {counts['train']:,} sequences -> {train_path}")
    print(f"  dev.txt:   {counts['dev']:,} sequences -> {dev_path}")
    print(f"  test.txt:  {counts['test']:,} sequences -> {test_path}")


def preprocess_hyena16k(fasta_path: str, out_dir: str):
    """HY16k-HRG: 16,384 bp windows without overlap; N is kept."""
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.txt")
    dev_path = os.path.join(out_dir, "dev.txt")

    train_count, dev_count = 0, 0
    with open(train_path, "w") as ftrain, open(dev_path, "w") as fdev:
        for chrom, raw_seq in parse_fasta(fasta_path):
            seq = clean_sequence(raw_seq)
            windows = chunk_windows(seq, 16384)
            print(f"  chr{chrom}: {len(seq):,} bp -> {len(windows):,} windows")
            for w in windows:
                if chrom == VAL_CHROMOSOME:
                    fdev.write(w + "\n")
                    dev_count += 1
                else:
                    ftrain.write(w + "\n")
                    train_count += 1

    print("HY16k-HRG done:")
    print(f"  train.txt: {train_count:,} sequences -> {train_path}")
    print(f"  dev.txt:   {dev_count:,} sequences -> {dev_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True,
                        help="Folder for the genome FASTA and the corpus folders")
    parser.add_argument("--corpora", nargs="+", default=CORPORA, choices=CORPORA,
                        help="Which corpora to write (default: all four)")
    args = parser.parse_args()

    fasta_path = download_fasta(args.output_dir)

    if "DB2-HRG" in args.corpora:
        print("\n=== DB2-HRG ===")
        preprocess_db2(fasta_path, os.path.join(args.output_dir, "DB2-HRG"))

    if "NT-HRG" in args.corpora:
        print("\n=== NT-HRG ===")
        preprocess_nt(fasta_path, os.path.join(args.output_dir, "NT-HRG"))

    if "HY-HRG" in args.corpora:
        print("\n=== HY-HRG ===")
        preprocess_hyena(fasta_path, os.path.join(args.output_dir, "HY-HRG"))

    if "HY16k-HRG" in args.corpora:
        print("\n=== HY16k-HRG ===")
        preprocess_hyena16k(fasta_path, os.path.join(args.output_dir, "HY16k-HRG"))

    print("\nAll done!")


if __name__ == "__main__":
    main()
