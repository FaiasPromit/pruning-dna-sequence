"""
BEND embedder for the Hyena-16k models pretrained in this repository
(hyenadna/pretraining, HyenaDNAForCausalLM with the single-character tokenizer).

Goes to: BEND/bend/utils/hyena16k_hrg_embedder.py

BEND's own HyenaDNAEmbedder only supports the five official HazyResearch checkpoints
(it looks up the length from the checkpoint name and loads the official model class).
These models are a different class saved with save_pretrained, so they need their own
embedder. It loads the model with the code in hyenadna/model_code and returns the output
of the backbone, one embedding per nucleotide, so no upsampling is needed for gene finding.

BEND calls an embedder on one sequence at a time as embed([sequence])[0] and then checks
that shape[1] equals the sequence length. Each returned array therefore keeps the batch
dimension, shape (1, L, d_model). Without it BEND reads d_model as the length, skips every
sequence, and later stops with "No samples found in dataset".
"""

import os
import sys
from typing import List

import numpy as np
import torch

from bend.utils.embedders import BaseEmbedder


class Hyena16kHRGEmbedder(BaseEmbedder):
    """Embedder for a Hyena-16k model saved by hyenadna/pretraining."""

    def load_model(
        self,
        model_path: str,
        model_code_dir: str,
        device: str = None,
        bf16: bool = True,
        **kwargs,  # takes the other settings of the embed.yaml block, such as upsample_embeddings
    ):
        """
        Parameters
        ----------
        model_path : str
            Model folder saved with save_pretrained (final_model or checkpoint-*),
            with config.json and pytorch_model.bin.
        model_code_dir : str
            Folder with modeling_hyenadna.py, configuration_hyenadna.py and
            tokenization_hyenadna.py (hyenadna/model_code).
        device : str, optional
            'cuda' or 'cpu'. Chosen automatically if None.
        bf16 : bool
            Run the model under bf16 autocast on CUDA, as in pretraining. The
            returned embeddings are float32.
        """
        if model_code_dir not in sys.path:
            sys.path.insert(0, model_code_dir)
        from modeling_hyenadna import HyenaDNAForCausalLM
        from configuration_hyenadna import HyenaDNAConfig
        from tokenization_hyenadna import CharacterTokenizer

        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.autocast_dtype = (
            torch.bfloat16 if (bf16 and self.device.type == "cuda") else torch.float32
        )

        config = HyenaDNAConfig.from_pretrained(model_path)
        self.model = HyenaDNAForCausalLM.from_pretrained(model_path, config=config)
        self.model.eval().to(self.device)

        # The same tokenizer as in pretraining: A, C, G, T, N -> 7 to 11, no special tokens
        self.tokenizer = CharacterTokenizer(
            characters=["A", "C", "G", "T", "N"],
            model_max_length=config.max_seq_length + 2,
        )
        self.embedding_dim = config.d_model
        self._shape_logged = False
        print(
            f"[Hyena16kHRGEmbedder] loaded {model_path} | "
            f"d_model={self.embedding_dim} | max_seq_length={config.max_seq_length} | "
            f"device={self.device} | autocast={self.autocast_dtype}",
            flush=True,
        )

    @torch.no_grad()
    def embed(
        self,
        sequences: List[str],
        disable_tqdm: bool = True,
        upsample_embeddings: bool = False,  # not needed: there is already one embedding per nucleotide
        **kwargs,
    ) -> List[np.ndarray]:
        """Return one float32 array of shape (1, L, d_model) per input sequence."""
        outputs = []
        for seq in sequences:
            # As in corpus preparation: upper case, anything other than A, C, G, T becomes N
            seq = seq.upper()
            seq = "".join(c if c in "ACGT" else "N" for c in seq)

            enc = self.tokenizer(seq, add_special_tokens=False, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)  # (1, L)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype,
                enabled=(self.autocast_dtype != torch.float32),
            ):
                hidden = self.model.backbone(input_ids)

            # Take the tensor if the backbone returns a tuple or a ModelOutput
            if isinstance(hidden, (tuple, list)):
                hidden = hidden[0]
            elif hasattr(hidden, "last_hidden_state"):
                hidden = hidden.last_hidden_state

            if hidden.dim() != 3:
                raise RuntimeError(
                    f"expected a 3D backbone output (B, L, D), got {tuple(hidden.shape)}"
                )

            # Make sure the layout is (1, L, d_model)
            if hidden.shape[1] == self.embedding_dim and hidden.shape[2] != self.embedding_dim:
                hidden = hidden.transpose(1, 2)

            if not self._shape_logged:
                print(
                    f"[Hyena16kHRGEmbedder] returning shape {tuple(hidden.shape)} "
                    f"for sequence length {len(seq)} (BEND needs shape[1] == length)",
                    flush=True,
                )
                self._shape_logged = True

            if hidden.shape[1] != len(seq):
                raise RuntimeError(
                    f"embedding length {hidden.shape[1]} != sequence length {len(seq)}; "
                    f"shape={tuple(hidden.shape)}"
                )

            outputs.append(hidden.float().cpu().numpy())  # (1, L, d_model)
        return outputs
