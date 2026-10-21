#!/usr/bin/env python
"""Write vocab.txt for the Nucleotide Transformer 6-mer tokenizer.

The vocabulary has 4,105 tokens:
  0 <unk>, 1 <pad>, 2 <mask>, 3 <cls>
  4 to 4099: all 4,096 6-mers (AAAAAA to GGGGGG)
  4100 to 4104: the single bases A, T, C, G and N

The vocab.txt in this folder was written by this script; running it again
gives the same file.

Usage:
    python generate_vocab.py --output_dir .
"""

import itertools, os, argparse

BASES = "ATCG"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    vocab = ["<unk>", "<pad>", "<mask>", "<cls>"]
    for kmer in itertools.product(BASES, repeat=6):
        vocab.append("".join(kmer))
    vocab.extend(["A", "T", "C", "G", "N"])

    out = os.path.join(args.output_dir, "vocab.txt")
    with open(out, "w") as f:
        for tok in vocab:
            f.write(tok + "\n")
    print(f"Wrote {len(vocab)} tokens to {out}")

if __name__ == "__main__":
    main()
