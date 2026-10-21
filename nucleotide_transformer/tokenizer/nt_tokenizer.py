"""
Nucleotide Transformer 6-mer tokenizer.

A DNA sequence is read from left to right in 6-mers. Where a 6-mer cannot be
formed (it contains N, or fewer than 6 bases are left), single-base tokens are
used instead.

Vocabulary (4,105 tokens, vocab.txt):
  0 <unk>   1 <pad>   2 <mask>   3 <cls>
  4 to 4099      all 4,096 6-mers (AAAAAA to GGGGGG)
  4100 to 4104   the single bases A, T, C, G and N

save_pretrained writes vocab.txt and the tokenizer config into a model folder,
together with a copy of this file, so the fine-tuning scripts can load the
tokenizer from the model folder.
"""

import os
from typing import Dict, List, Optional


class NTTokenizer:
    """Lightweight 6-mer tokenizer for Nucleotide Transformer."""

    # Fixed special token ids
    unk_token_id = 0
    pad_token_id = 1
    mask_token_id = 2
    cls_token_id = 3

    def __init__(self, vocab_file: str, model_max_length: int = 1000):
        self.model_max_length = model_max_length  # maximum number of tokens, not counting <cls>
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        with open(vocab_file) as f:
            for idx, line in enumerate(f):
                tok = line.strip()
                self.vocab[tok] = idx
                self.id_to_token[idx] = tok

        self.vocab_size = len(self.vocab)
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"
        self.cls_token = "<cls>"
        self.unk_token = "<unk>"

    def __len__(self):
        return self.vocab_size

    # Tokenization
    def tokenize_sequence(self, seq: str, max_tokens: Optional[int] = None) -> List[int]:
        """Convert a raw DNA string to a list of token ids (no CLS, no padding)."""
        if max_tokens is None:
            max_tokens = self.model_max_length
        seq = seq.upper()
        ids: List[int] = []
        i = 0
        while i < len(seq) and len(ids) < max_tokens:
            if i + 6 <= len(seq):
                kmer = seq[i : i + 6]
                if kmer in self.vocab:
                    ids.append(self.vocab[kmer])
                    i += 6
                    continue
            # fallback: single nucleotide
            c = seq[i]
            ids.append(self.vocab.get(c, self.unk_token_id))
            i += 1
        return ids

    def encode(
        self,
        sequence: str,
        max_length: Optional[int] = None,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ) -> dict:
        """Tokenize one sequence into a dict with 'input_ids' and 'attention_mask'.

        Adds <cls> at position 0 and pads or truncates to max_length tokens in total.
        """
        if max_length is None:
            max_length = self.model_max_length + 1  # +1 for CLS

        content = self.tokenize_sequence(sequence, max_tokens=max_length - 1)
        input_ids = [self.cls_token_id] + content

        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        attention_mask = [1] * len(input_ids)

        if padding == "max_length":
            pad_len = max_length - len(input_ids)
            input_ids += [self.pad_token_id] * pad_len
            attention_mask += [0] * pad_len

        out = {"input_ids": input_ids, "attention_mask": attention_mask}

        if return_tensors == "pt":
            import torch
            out = {k: torch.tensor([v]) for k, v in out.items()}

        return out

    def batch_encode(
        self,
        sequences: List[str],
        max_length: Optional[int] = None,
        padding: str = "max_length",
    ) -> dict:
        """Encode a list of sequences."""
        results = [self.encode(s, max_length=max_length, padding=padding) for s in sequences]
        return {
            "input_ids": [r["input_ids"] for r in results],
            "attention_mask": [r["attention_mask"] for r in results],
        }

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.id_to_token.get(i, "<unk>") for i in ids)

    # Saving and loading, in the style of Hugging Face tokenizers
    def save_pretrained(self, save_directory: str):
        """Save vocab and tokenizer config into save_directory."""
        os.makedirs(save_directory, exist_ok=True)

        # vocab
        vocab_dst = os.path.join(save_directory, "vocab.txt")
        with open(vocab_dst, "w") as f:
            for idx in range(self.vocab_size):
                f.write(self.id_to_token[idx] + "\n")

        # tokenizer_config.json
        import json
        cfg = {
            "tokenizer_class": "NTTokenizer",
            "model_max_length": self.model_max_length,
        }
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        # special_tokens_map.json
        stm = {
            "cls_token": "<cls>",
            "mask_token": "<mask>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
        }
        with open(os.path.join(save_directory, "special_tokens_map.json"), "w") as f:
            json.dump(stm, f, indent=2)

        # Copy this file so the fine-tuning scripts can import it from the model folder
        import shutil
        src = os.path.abspath(__file__)
        dst = os.path.join(save_directory, "nt_tokenizer.py")
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        vocab_file = os.path.join(path, "vocab.txt")
        import json
        cfg_path = os.path.join(path, "tokenizer_config.json")
        max_len = 1000
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            max_len = cfg.get("model_max_length", 1000)
        max_len = kwargs.get("model_max_length", max_len)
        return cls(vocab_file=vocab_file, model_max_length=max_len)
