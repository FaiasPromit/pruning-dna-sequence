"""
Train a DNABERT-2 style BPE tokenizer on a DNA corpus.

The tokenizer uses a whitespace pre-tokenizer and five special tokens with the
same IDs as DNABERT-2: [UNK]=0, [CLS]=1, [SEP]=2, [PAD]=3, [MASK]=4. Every
encoded sequence is wrapped as [CLS] ... [SEP]. The result is saved in the
Hugging Face tokenizers JSON format.

Usage:
    python train_bpe_tokenizer.py --corpus-files train.txt dev.txt \
        --vocab-size 4096 --min-frequency 2 --save-path DB2-MSG/tokenizer.json
"""
from tokenizers import Tokenizer, trainers, models, pre_tokenizers, processors


def train_tokenizer(
    files,
    vocab_size=4096,
    min_frequency=2,
    special_tokens=("[UNK]","[CLS]","[SEP]","[PAD]","[MASK]"),
    save_path="tokenizer.json"
):
    # Count total lines for reference
    print("Counting lines in corpus files...")
    total_lines = 0
    for file_path in files:
        with open(file_path, 'r') as f:
            total_lines += sum(1 for _ in f)
    print(f"Total lines to process: {total_lines:,}")
    print(f"Target vocab size: {vocab_size}")
    print(f"Starting BPE training... (this may take a while)")

    # BPE model with [UNK] as the unknown token, as in DNABERT-2
    bpe = models.BPE(unk_token=special_tokens[0], fuse_unk=False)
    tokenizer = Tokenizer(bpe)

    # Whitespace pre-tokenizer, with no normalizer or decoder, as in DNABERT-2
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Special tokens receive IDs in the order listed:
    # [UNK]=0, [CLS]=1, [SEP]=2, [PAD]=3, [MASK]=4
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(special_tokens)
    )

    paths = [str(p) for p in files]
    tokenizer.train(files=paths, trainer=trainer)

    # Reload from JSON, then add the post-processor
    tok_json = tokenizer.to_str()
    tok = Tokenizer.from_str(tok_json)

    # Wrap a single sequence as [CLS] A [SEP] and a pair as [CLS] A [SEP] B [SEP],
    # with the second sequence of a pair given token type 1
    tok.post_processor = processors.TemplateProcessing(
        single=f"{special_tokens[1]} $A {special_tokens[2]}",
        pair=f"{special_tokens[1]} $A {special_tokens[2]} $B:1 {special_tokens[2]}:1",
        special_tokens=[
            (special_tokens[1], tok.token_to_id(special_tokens[1])),
            (special_tokens[2], tok.token_to_id(special_tokens[2])),
        ],
    )

    tok.save(save_path)
    print(f"Saved DNABERT2-style tokenizer to {save_path}")


if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--corpus-files", nargs="+", required=True)
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--save-path", type=str, default="tokenizer.json")
    args = p.parse_args()

    files = [Path(f) for f in args.corpus_files]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        print("ERROR: These files do not exist:")
        for m in missing:
            print(" -", m)
        sys.exit(1)

    train_tokenizer(
        files=files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        save_path=args.save_path
    )
