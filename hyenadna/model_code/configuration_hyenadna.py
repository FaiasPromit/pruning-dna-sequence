#!/usr/bin/env python
"""
HyenaDNAConfig: a Hugging Face PretrainedConfig for the HyenaDNA model in
modeling_hyenadna.py, so a model saves with save_pretrained (config.json plus
pytorch_model.bin) and loads with from_pretrained.

The defaults are the 8-layer, width-256 HyenaDNA architecture used in the paper.
max_seq_length is the fixed window length used in pretraining (1,024 for Hyena,
16,384 for Hyena-16k); there is no sequence-length warmup.
"""

from transformers.configuration_utils import PretrainedConfig


class HyenaDNAConfig(PretrainedConfig):
    model_type = "hyenadna"

    def __init__(
        self,
        vocab_size: int = 12,          # 7 special tokens + A, C, G, T, N
        d_model: int = 256,            # width
        n_layer: int = 8,              # depth
        d_inner: int = 1024,           # MLP hidden size (4 * d_model)
        max_seq_length: int = 1024,    # nucleotides per training window
        # Hyena operator and implicit filter
        order: int = 2,                # Hyena recurrence depth
        filter_order: int = 64,        # width of the implicit filter MLP
        filter_emb_dim: int = 5,       # positional encoding size (odd, >= 3)
        filter_w: float = 10.0,        # frequency of the Sin activation
        filter_lr: float = 6e-4,       # learning rate stored on the filter parameters (not used by our optimizer)
        filter_wd: float = 0.0,        # weight decay stored on the filter parameters
        lr_pos_emb: float = 1e-5,      # learning rate stored on the positional embedding
        modulate: bool = True,
        # Regularization and other settings
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = False,
        pad_vocab_size_multiple: int = 1,
        initializer_range: float = 0.02,
        gradient_checkpointing: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if pad_vocab_size_multiple > 1 and vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_inner = d_inner
        self.max_seq_length = max_seq_length

        self.order = order
        self.filter_order = filter_order
        self.filter_emb_dim = filter_emb_dim
        self.filter_w = filter_w
        self.filter_lr = filter_lr
        self.filter_wd = filter_wd
        self.lr_pos_emb = lr_pos_emb
        self.modulate = modulate

        self.resid_dropout = resid_dropout
        self.embed_dropout = embed_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.initializer_range = initializer_range
        self.gradient_checkpointing = gradient_checkpointing

        # Hyena has no position embeddings; position enters through the implicit filter
        self.max_position_embeddings = 0

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def l_max(self) -> int:
        # Longest token sequence the filters are built for: the window plus room for
        # 2 special tokens. Shorter inputs use the first part of the filter.
        return self.max_seq_length + 2

    def build_layer_dict(self) -> dict:
        """Arguments passed to HyenaOperator and HyenaFilter."""
        return {
            "d_model": self.d_model,
            "l_max": self.l_max,
            "order": self.order,
            "filter_order": self.filter_order,
            "dropout": self.resid_dropout,
            "filter_dropout": 0.0,
            # passed on to HyenaFilter
            "emb_dim": self.filter_emb_dim,
            "w": self.filter_w,
            "lr": self.filter_lr,
            "wd": self.filter_wd,
            "lr_pos_emb": self.lr_pos_emb,
            "modulate": self.modulate,
        }
