#!/usr/bin/env python
"""
Single-character DNA tokenizer, as in HazyResearch/hyena-dna.

Vocabulary (12 ids):
    [CLS]=0  [SEP]=1  [BOS]=2  [MASK]=3  [PAD]=4  [RESERVED]=5  [UNK]=6
    A=7  C=8  G=9  T=10  N=11

N is a normal token, so the HY-HRG corpus keeps ambiguous bases (the DNABERT-2
corpora drop them).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(self, characters: Sequence[str], model_max_length: int,
                 padding_side: str = 'right', **kwargs):
        self.characters = characters
        self.model_max_length = model_max_length

        # The vocabulary is built before super().__init__(), because newer
        # transformers versions call get_vocab() while adding the special tokens
        self._vocab_str_to_int = {
            "[CLS]": 0, "[SEP]": 1, "[BOS]": 2, "[MASK]": 3,
            "[PAD]": 4, "[RESERVED]": 5, "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)
        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        super().__init__(
            bos_token=bos_token,
            eos_token=sep_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None,
                                already_has_special_tokens: bool = False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1,
                already_has_special_tokens=True)
        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizer":
        cfg = {"characters": [chr(i) for i in config["char_ords"]],
               "model_max_length": config["model_max_length"]}
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file, "w") as f:
            json.dump(self.get_config(), f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)


# The tokenizer used by the pretraining and fine-tuning scripts
DNA_CHARACTERS = ["A", "C", "G", "T", "N"]


def build_dna_tokenizer(model_max_length: int) -> CharacterTokenizer:
    return CharacterTokenizer(characters=DNA_CHARACTERS, model_max_length=model_max_length)
