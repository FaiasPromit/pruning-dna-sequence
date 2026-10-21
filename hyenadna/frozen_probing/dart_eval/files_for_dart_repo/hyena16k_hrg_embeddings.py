"""
DART-Eval embedding extractor for the Hyena-16k models pretrained in this repository
(hyenadna/pretraining, HyenaDNAForCausalLM with the single-character tokenizer).

Goes to: DART-Eval/src/dnalm_bench/task_2_5_single/hyena16k_hrg_embeddings.py

DART-Eval's own HyenaDNAEmbeddingExtractor loads the official checkpoints from the
LongSafari Hugging Face repositories with AutoModelForCausalLM, so it cannot load these
models: they are a different class saved with save_pretrained, and their forward pass
does not return hidden states. This extractor loads the model with the code in
hyenadna/model_code and reads the output of the backbone directly.

The model folder and the model code folder come from environment variables, so the same
files serve every pretraining variant (see dart_task4.sbatch):
    HYENA16K_MODEL_DIR   model folder saved by the pretraining script (final_model or checkpoint-*)
    HYENA16K_CODE_DIR    folder with modeling_hyenadna.py, configuration_hyenadna.py and
                         tokenization_hyenadna.py (hyenadna/model_code)
"""

import os
import sys

import numpy as np
import torch

from .embeddings import HFEmbeddingExtractor, SimpleEmbeddingExtractor
from ..utils import onehot_to_chars


class Hyena16kHRGEmbeddingExtractor(HFEmbeddingExtractor, SimpleEmbeddingExtractor):
    # One token per nucleotide, so embedding i belongs to nucleotide i and the
    # positions are a fixed [0, L] slice
    _idx_mode = "fixed"

    def __init__(self, model_path, model_code_dir, batch_size, num_workers, device):
        if model_code_dir not in sys.path:
            sys.path.insert(0, model_code_dir)
        from modeling_hyenadna import HyenaDNAForCausalLM
        from configuration_hyenadna import HyenaDNAConfig
        from tokenization_hyenadna import CharacterTokenizer

        config = HyenaDNAConfig.from_pretrained(model_path)
        model = HyenaDNAForCausalLM.from_pretrained(model_path, config=config)
        tokenizer = CharacterTokenizer(
            characters=["A", "C", "G", "T", "N"],
            model_max_length=config.max_seq_length + 2,
            padding_side="right",
        )
        self.embedding_dim = config.d_model
        self._shape_logged = False
        print(
            f"[Hyena16kHRG] loaded {model_path} | d_model={config.d_model} "
            f"| max_seq_length={config.max_seq_length}",
            flush=True,
        )
        super().__init__(tokenizer, model, batch_size, num_workers, device)

    def tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer(
            seqs_str, return_tensors="pt", padding=True, add_special_tokens=False
        )
        return encoded["input_ids"], None

    def model_fwd(self, tokens):
        """Return the backbone output, shape (batch, length, d_model).

        The model has no output_hidden_states option, so the backbone is called directly.
        """
        tokens = tokens.to(device=self.device)
        with torch.no_grad():
            embs = self.model.backbone(tokens)
        if isinstance(embs, (tuple, list)):
            embs = embs[0]
        elif hasattr(embs, "last_hidden_state"):
            embs = embs.last_hidden_state

        # Make sure the layout is (batch, length, d_model)
        if embs.dim() == 3 and embs.shape[1] == self.embedding_dim and embs.shape[2] != self.embedding_dim:
            embs = embs.transpose(1, 2)

        if not self._shape_logged:
            print(f"[Hyena16kHRG] embedding batch shape {tuple(embs.shape)}", flush=True)
            self._shape_logged = True
        return embs

    @staticmethod
    def _offsets_to_indices(offsets, seqs):
        # One embedding per nucleotide over the whole sequence
        return np.array([0, seqs.shape[1]])
