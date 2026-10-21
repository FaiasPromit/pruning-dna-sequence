#!/usr/bin/env python
"""
HyenaDNA model, packaged as a Hugging Face PreTrainedModel.

The Hyena operator, implicit filter, FFT long convolution, block and backbone
follow standalone_hyenadna.py from HazyResearch/hyena-dna and compute the same
thing. The changes are packaging only:

  - The model is a transformers PreTrainedModel with HyenaDNAConfig, so it saves
    with save_pretrained(safe_serialization=False) as config.json plus
    pytorch_model.bin, the same layout as the DNABERT-2 and NT-500M models.
  - HyenaDNAForCausalLM is the next-token prediction model used for pretraining.
  - HyenaDNAForSequenceClassification adds a mean-pooling and linear head for
    fine-tuning on classification tasks.
  - The Hyena mixer always runs in float32, because torch.fft has no half
    precision CUDA support and the long convolution is unstable in bf16. The MLP
    and embeddings still run in bf16 under autocast.
  - StochasticDepth is a small local copy instead of the torchvision one, and the
    attention path of the original code is left out (the model is pure Hyena).
  - For B-PAS (hyenadna/pretraining/run_clm_bpas.py), the backbone can record the
    standard deviation of its hidden state after every Hyena block. This is off
    unless the training script sets _collect_layer_stds = True, and it does not
    change the model's output.

The pruning logic lives in the pretraining scripts, not here.
"""

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    CausalLMOutput,
    SequenceClassifierOutput,
)

from configuration_hyenadna import HyenaDNAConfig


# Hyena operator (as in HazyResearch/hyena-dna)

def fftconv(u, k, D):
    """Convolution via the Fourier domain (Convolution Theorem)."""
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


class OptimModule(nn.Module):
    """Register buffers/parameters with configurable optimizer hyperparameters."""

    def register(self, name, tensor, lr=None, wd=0.0):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {}
            if lr is not None:
                optim["lr"] = lr
            if wd is not None:
                optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)


class Sin(nn.Module):
    """Sin activation for the Hyena filter MLP."""

    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()
        self.seq_len = seq_len
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(OptimModule):
    """Window function applied to the output of the (MLP) filter."""

    def __init__(self, d_model, fast_decay_pct=0.3, slow_decay_pct=1.5, target=1e-2,
                 modulation_lr=0.0, modulate: bool = True, shift: float = 0.05, **kwargs):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class HyenaFilter(OptimModule):
    def __init__(self, d_model, emb_dim=3, order=16, fused_fft_conv=False, seq_len=1024,
                 lr=1e-3, lr_pos_emb=1e-5, dropout=0.0, w=1, wd=0, bias=True,
                 num_inner_mlps=2, normalized=False, **kwargs):
        """Implicit long filter with modulation."""
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, \
            "emb_dim must be odd and >= 3 (time, sine and cosine)"
        self.seq_len = seq_len

        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        # Implicit filter MLP. The original code uses one Sin activation (one freq
        # parameter) between all its layers. That is kept here, but the activation
        # is applied in filter() instead of being repeated inside an nn.Sequential,
        # so the shared parameter appears once in the state dict and
        # save_pretrained does not see duplicate tensors.
        self.filter_act = act
        self.num_inner_mlps = num_inner_mlps
        linears = [nn.Linear(emb_dim, order)]
        for _ in range(num_inner_mlps):
            linears.append(nn.Linear(order, order))
        linears.append(nn.Linear(order, d_model, bias=False))
        self.implicit_filter = nn.ModuleList(linears)

        self.modulation = ExponentialModulation(d_model, **kwargs)
        self.normalized = normalized

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = z
        n = len(self.implicit_filter)
        for i, lin in enumerate(self.implicit_filter):
            h = lin(h)
            if i < n - 1:                      # Sin after every layer except the last one
                h = self.filter_act(h)
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None:
            k = self.filter(L)
        k = k[0] if type(k) is tuple else k
        y = fftconv(x, k, bias)
        return y


class HyenaOperator(nn.Module):
    def __init__(self, d_model, l_max, order=2, filter_order=64, dropout=0.0,
                 filter_dropout=0.0, **filter_args):
        """Hyena operator (https://arxiv.org/pdf/2302.10866.pdf)."""
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)

        self.short_filter = nn.Conv1d(inner_width, inner_width, 3, padding=2, groups=inner_width)
        self.filter_fn = HyenaFilter(
            d_model * (order - 1), order=filter_order, seq_len=l_max,
            channels=1, dropout=filter_dropout, **filter_args,
        )

    def forward(self, u, *args, **kwargs):
        # The Hyena mixer runs in float32: torch.fft has no half precision CUDA
        # kernel and the long convolution is unstable in bf16. The MLP and
        # embeddings around it still run in bf16 under autocast.
        orig_dtype = u.dtype
        with torch.autocast(device_type=u.device.type, enabled=False):
            u = u.float()
            l = u.size(-2)
            l_filter = min(l, self.l_max)
            u = self.in_proj(u)
            u = rearrange(u, 'b l d -> b d l')

            uc = self.short_filter(u)[..., :l_filter]
            *x, v = uc.split(self.d_model, dim=1)

            k = self.filter_fn.filter(l_filter)[0]
            k = rearrange(k, 'l (o d) -> o d l', o=self.order - 1)
            bias = rearrange(self.filter_fn.bias, '(o d) -> o d', o=self.order - 1)

            for o, x_i in enumerate(reversed(x[1:])):
                v = self.dropout(v * x_i)
                v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

            y = rearrange(v * x[0], 'b d l -> b l d')
            y = self.out_proj(y)
        return y.to(orig_dtype)


# MLP, block and backbone (as in HazyResearch/hyena-dna, without the attention path)

class StochasticDepth(nn.Module):
    """Drop-path, a small copy of torchvision.ops.StochasticDepth (mode='row')."""

    def __init__(self, p: float = 0.0, mode: str = "row"):
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        survival = 1.0 - self.p
        if self.mode == "row":
            shape = [x.shape[0]] + [1] * (x.ndim - 1)
        else:
            shape = [1] * x.ndim
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(survival)
        return x * mask / survival


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 activation=F.gelu, return_residual=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


class Block(nn.Module):
    def __init__(self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm,
                 dropout_cls=nn.Dropout, prenorm=True, resid_dropout1=0., resid_dropout2=0.,
                 drop_path1=0., drop_path2=0., return_residual=False, residual_in_fp32=False):
        """Pre-norm block: Dropout -> Add -> LayerNorm -> Mixer -> Dropout -> Add -> LayerNorm -> MLP."""
        super().__init__()
        self.prenorm = prenorm
        self.return_residual = return_residual
        self.residual_in_fp32 = residual_in_fp32
        if self.residual_in_fp32:
            assert self.prenorm, 'residual_in_fp32 is only compatible with prenorm=True'
        self.mixer = mixer_cls()
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode='row')
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = StochasticDepth(drop_path2, mode='row')
            self.norm2 = norm_cls(dim)

    def forward(self, hidden_states, residual=None, mixer_subset=None, mixer_kwargs=None):
        dropped = self.drop_path1(self.dropout1(hidden_states))
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, **(mixer_kwargs if mixer_kwargs is not None else {}))

        if not isinstance(self.mlp, nn.Identity):
            dropped = self.drop_path2(self.dropout2(hidden_states))
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


def create_block(d_model, d_inner, layer, layer_norm_epsilon=1e-5,
                 resid_dropout1=0.0, resid_dropout2=0.0, residual_in_fp32=False, layer_idx=None):
    mixer_cls = partial(HyenaOperator, **layer)
    mlp_cls = partial(Mlp, hidden_features=d_inner, activation=partial(F.gelu, approximate='tanh'))
    norm_cls = partial(nn.LayerNorm, eps=layer_norm_epsilon)
    block = Block(d_model, mixer_cls, mlp_cls, norm_cls=norm_cls, prenorm=True,
                  resid_dropout1=resid_dropout1, resid_dropout2=resid_dropout2,
                  residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block


def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True,
                  glu_act=False):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
            elif name in ["output_linear.0.weight"]:
                if not glu_act:
                    nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
                else:
                    out_features = p.shape[0]
                    nn.init.normal_(p[:out_features // 2], mean=0.0,
                                    std=initializer_range / math.sqrt(2 * n_layer) * 2)


class GPT2Embeddings(nn.Module):
    def __init__(self, embed_dim, vocab_size, max_position_embeddings, padding_idx=None):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.project_in = None
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim)

    def forward(self, input_ids, position_ids=None):
        _batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            embeddings = embeddings + self.position_embeddings(position_ids)
        return embeddings


class HyenaDNABackbone(nn.Module):
    """Stack of Hyena blocks (LMBackbone in hyena-dna). Hyena uses no position embeddings."""

    def __init__(self, config: HyenaDNAConfig):
        super().__init__()
        self.gradient_checkpointing = config.gradient_checkpointing
        # B-PAS: when the training script sets _collect_layer_stds = True, every
        # forward pass stores the std of the hidden state after each Hyena block in
        # _layer_stds (one number per block). Off by default.
        self._collect_layer_stds = False
        self._layer_stds = []
        layer = config.build_layer_dict()

        self.embeddings = GPT2Embeddings(config.d_model, config.vocab_size,
                                         config.max_position_embeddings)
        self.layers = nn.ModuleList([
            create_block(
                config.d_model, d_inner=config.d_inner, layer=layer,
                layer_norm_epsilon=config.layer_norm_epsilon,
                resid_dropout1=config.embed_dropout if i == 0 else config.resid_dropout,
                resid_dropout2=config.resid_dropout,
                residual_in_fp32=config.residual_in_fp32, layer_idx=i,
            ) for i in range(config.n_layer)
        ])
        self.drop_f = nn.Dropout(config.resid_dropout)
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.apply(partial(_init_weights, n_layer=config.n_layer))

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        residual = None
        # B-PAS: start a new list for this forward pass
        if self._collect_layer_stds:
            self._layer_stds = []
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states, residual = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, residual, use_reentrant=False,
                )
            else:
                hidden_states, residual = layer(hidden_states, residual)
            # B-PAS: hidden_states + residual is what the next block (or the final
            # LayerNorm) receives, so it is the hidden state after this block, like
            # the encoder layer output whose std DNABERT-2 records. Computed in
            # float32 and without gradients.
            if self._collect_layer_stds:
                with torch.no_grad():
                    rep = hidden_states if residual is None else (hidden_states + residual)
                    if rep.numel() > 1:
                        self._layer_stds.append(torch.std(rep.float()).item())
        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
        return hidden_states


# Hugging Face PreTrainedModel classes

class HyenaDNAPreTrainedModel(PreTrainedModel):
    config_class = HyenaDNAConfig
    base_model_prefix = "hyenadna"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        # The backbone already applies the GPT-2 style initialization in its
        # constructor; this covers the heads added on top of it.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)


class HyenaDNAForCausalLM(HyenaDNAPreTrainedModel):
    """Next-token-prediction head for pretraining."""

    def __init__(self, config: HyenaDNAConfig):
        super().__init__(config)
        self.backbone = HyenaDNABackbone(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.backbone.embeddings.word_embeddings = value

    def forward(self, input_ids=None, labels=None, position_ids=None, **kwargs):
        hidden_states = self.backbone(input_ids, position_ids=position_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Causal language modeling: predict token t+1 from tokens up to t
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits)


class HyenaDNAForSequenceClassification(HyenaDNAPreTrainedModel):
    """Mean pooling over the sequence and a linear classification head (the 'pool' head of HyenaDNA)."""

    def __init__(self, config: HyenaDNAConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.backbone = HyenaDNABackbone(config)
        self.score = nn.Linear(config.d_model, config.num_labels)
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.embeddings.word_embeddings

    def forward(self, input_ids=None, attention_mask=None, labels=None, position_ids=None, **kwargs):
        hidden_states = self.backbone(input_ids, position_ids=position_ids)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = hidden_states.mean(dim=1)
        logits = self.score(pooled)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.squeeze(-1).float(), labels.float())
            else:
                loss = F.cross_entropy(logits.float(), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits)
