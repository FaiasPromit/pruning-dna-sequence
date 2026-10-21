#!/usr/bin/env python
"""
InfoBatch (Qin et al., 2024, arXiv:2303.04947) for DNABERT-2 MLM pretraining,
with each sequence scored by its mean MLM loss. Equation numbers below refer to
Qin et al.

  1. Soft pruning (Eq. 3): the threshold is the mean of all current scores.
     Sequences below it are skipped with probability r; the rest are always
     kept. The draw is repeated every epoch, so nothing is removed permanently.
  2. Expectation rescaling: kept sequences from the below-mean group have
     their loss scaled by 1 / (1 - r), so the expected gradient matches
     training on the full corpus.
  3. Annealing (Eq. 10): pruning runs only for the first delta * C epochs of
     the C-epoch budget; the remaining epochs use the full corpus without
     rescaling.

A score is the sequence's mean MLM cross-entropy in the last epoch it was
processed (Eq. 4). All scores start at 1 before the first epoch, and a
sequence that is skipped keeps its previous score.

Per-sequence statistics are kept in GPU tensors of length N and summed across
ranks once per epoch. Every rank makes the same keep/rescale decision because
the random draw is seeded from (seed, epoch).
"""

import json
import logging
import math
import os

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Markers in the reported accuracy array (the same values as in run_mlm.py)
ACC_NOT_SEEN = -1.0
ACC_NO_MASK = -2.0

# Rows of the per-sequence statistics tensor
_ROW_LOSS_SUM = 0   # sum of per-token MLM loss, grouped by sequence
_ROW_LOSS_CNT = 1   # number of loss-bearing tokens (labels != -100)
_ROW_ACC_CORRECT = 2  # correct predictions at [MASK] positions
_ROW_ACC_MASKED = 3   # number of [MASK] positions
_ROW_SEEN = 4         # times the sequence appeared in a batch
_N_ROWS = 5


class InfoBatch:
    """Per-epoch soft pruning + expectation rescaling over a fixed dataset."""

    def __init__(self, num_samples, r=0.5, delta=0.875, total_epochs=None,
                 seed=42, device=None, enabled=True):
        if not (0.0 < r < 1.0):
            raise ValueError(f"infobatch_r must be in (0,1), got {r}")
        if not (0.0 < delta <= 1.0):
            raise ValueError(f"infobatch_delta must be in (0,1], got {delta}")
        if total_epochs is None or total_epochs < 1:
            raise ValueError(
                "infobatch_total_epochs must be set (>= 1), because annealing "
                "starts at a fraction of the total number of epochs."
            )

        self.num_samples = int(num_samples)
        self.r = float(r)
        self.delta = float(delta)
        self.total_epochs = int(total_epochs)
        self.seed = int(seed)
        self.enabled = bool(enabled)
        self.device = device

        # All scores start at 1 before the first epoch
        self.scores = np.ones(self.num_samples, dtype=np.float32)
        # True once a sequence has ever contributed a real loss value.
        self.ever_scored = np.zeros(self.num_samples, dtype=bool)

        # Per-epoch accumulators (GPU)
        self._stats = torch.zeros(_N_ROWS, self.num_samples,
                                  dtype=torch.float32, device=device)
        # Which sequences get the 1 / (1 - r) loss weight this epoch (GPU)
        self._rescale_gpu = torch.zeros(self.num_samples, dtype=torch.bool,
                                        device=device)

        self.last_epoch_stats = {}
        self.last_rescale_mask = np.zeros(self.num_samples, dtype=bool)


    @property
    def anneal_start_epoch(self):
        """First epoch (counted from 0) with no pruning (Eq. 10: t >= delta * C)."""
        return int(math.floor(self.delta * self.total_epochs))

    def is_annealing(self, epoch):
        """epoch is counted from 0."""
        return (not self.enabled) or (epoch >= self.anneal_start_epoch)

    @property
    def scale(self):
        return 1.0 / (1.0 - self.r)


    def build_epoch(self, epoch):
        """Decide which sequences to train on this epoch and which to rescale.

        Returns (kept_indices, stats). The result depends only on (seed, epoch),
        so every rank computes the same subset without communication.
        """
        annealing = self.is_annealing(epoch)
        mean_score = float(self.scores.mean())

        if annealing:
            kept = np.arange(self.num_samples, dtype=np.int64)
            rescale = np.zeros(self.num_samples, dtype=bool)
            below = self.scores < mean_score
            n_dropped = 0
        else:
            below = self.scores < mean_score                      # Eq. 3
            rng = np.random.RandomState((self.seed * 1_000_003 + epoch) % (2**31 - 1))
            coin = rng.random_sample(self.num_samples) < self.r
            drop = below & coin
            keep_mask = ~drop
            rescale = below & keep_mask   # kept sequences from the below-mean group
            kept = np.where(keep_mask)[0].astype(np.int64)
            n_dropped = int(drop.sum())

        # Copy the rescale flags to the GPU for the training loop
        self._rescale_gpu.copy_(
            torch.from_numpy(rescale).to(self._rescale_gpu.device, non_blocking=True)
        )
        # CPU copy for the per-epoch reports and spreadsheets
        self.last_rescale_mask = rescale

        stats = {
            "epoch": epoch,
            "epoch_1indexed": epoch + 1,
            "annealing": annealing,
            "anneal_start_epoch_1indexed": self.anneal_start_epoch + 1,
            "mean_score": mean_score,
            "n_total": self.num_samples,
            "n_below_mean": int(below.sum()),
            "n_above_mean": int(self.num_samples - below.sum()),
            "pct_below_mean": 100.0 * float(below.sum()) / max(1, self.num_samples),
            "n_dropped": n_dropped,
            "n_kept": int(len(kept)),
            "n_rescaled": int(rescale.sum()),
            "effective_prune_pct": 100.0 * n_dropped / max(1, self.num_samples),
            "rescale_factor": 1.0 if annealing else self.scale,
            "r": self.r,
            "delta": self.delta,
            "total_epochs": self.total_epochs,
        }
        self.last_epoch_stats = stats
        return kept, stats

    def batch_weights(self, line_ids):
        """Per-sequence loss weights for a batch (line_ids is a LongTensor on the GPU)."""
        if self.is_annealing_current:
            return torch.ones(line_ids.shape[0], dtype=torch.float32,
                              device=line_ids.device)
        flags = self._rescale_gpu[line_ids]
        return torch.where(
            flags,
            torch.full_like(flags, self.scale, dtype=torch.float32),
            torch.ones(1, dtype=torch.float32, device=line_ids.device),
        )

    @property
    def is_annealing_current(self):
        return bool(self.last_epoch_stats.get("annealing", False))


    def reset_epoch_stats(self):
        self._stats.zero_()

    @torch.no_grad()
    def accumulate(self, line_ids, loss_sum_per_seq, loss_cnt_per_seq,
                   acc_correct_per_seq, acc_masked_per_seq):
        """Add one batch to the per-sequence sums. All arguments are 1-D GPU tensors of length batch_size."""
        ones = torch.ones_like(loss_cnt_per_seq)
        self._stats[_ROW_LOSS_SUM].index_add_(0, line_ids, loss_sum_per_seq.float())
        self._stats[_ROW_LOSS_CNT].index_add_(0, line_ids, loss_cnt_per_seq.float())
        self._stats[_ROW_ACC_CORRECT].index_add_(0, line_ids, acc_correct_per_seq.float())
        self._stats[_ROW_ACC_MASKED].index_add_(0, line_ids, acc_masked_per_seq.float())
        self._stats[_ROW_SEEN].index_add_(0, line_ids, ones.float())


    @torch.no_grad()
    def sync_and_update(self):
        """Sum the epoch's statistics across ranks and update the scores (Eq. 4).

        Returns (acc_array, update_info). acc_array holds each sequence's MLM
        accuracy this epoch, with ACC_NOT_SEEN and ACC_NO_MASK as markers.
        """
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self._stats, op=dist.ReduceOp.SUM)

        stats = self._stats.cpu().numpy()
        loss_sum = stats[_ROW_LOSS_SUM]
        loss_cnt = stats[_ROW_LOSS_CNT]
        acc_correct = stats[_ROW_ACC_CORRECT]
        acc_masked = stats[_ROW_ACC_MASKED]
        seen = stats[_ROW_SEEN]

        # Score update: new score = this epoch's mean loss for processed sequences, else unchanged
        has_loss = loss_cnt > 0
        new_scores = self.scores.copy()
        new_scores[has_loss] = (loss_sum[has_loss] / loss_cnt[has_loss]).astype(np.float32)

        n_updated = int(has_loss.sum())
        prev_mean = float(self.scores.mean())
        self.scores = new_scores
        self.ever_scored |= has_loss

        # Accuracy array (for reporting only)
        acc_array = np.full(self.num_samples, ACC_NOT_SEEN, dtype=np.float32)
        seen_mask = seen > 0
        no_mask = seen_mask & (acc_masked == 0)
        scored = seen_mask & (acc_masked > 0)
        acc_array[no_mask] = ACC_NO_MASK
        acc_array[scored] = acc_correct[scored] / acc_masked[scored]

        update_info = {
            "n_sequences_trained": int(seen_mask.sum()),
            "n_scores_updated": n_updated,
            "n_never_scored": int((~self.ever_scored).sum()),
            "score_mean_before": prev_mean,
            "score_mean_after": float(self.scores.mean()),
            "score_min": float(self.scores.min()),
            "score_max": float(self.scores.max()),
            "score_median": float(np.median(self.scores)),
            "epoch_loss_mean_over_trained": (
                float(loss_sum[has_loss].sum() / max(1.0, loss_cnt[has_loss].sum()))
                if n_updated else float("nan")
            ),
        }
        return acc_array, update_info


    def save(self, ckpt_dir):
        np.save(os.path.join(ckpt_dir, "infobatch_scores.npy"), self.scores)
        np.save(os.path.join(ckpt_dir, "infobatch_ever_scored.npy"), self.ever_scored)
        with open(os.path.join(ckpt_dir, "infobatch_state.json"), "w") as f:
            json.dump({
                "r": self.r,
                "delta": self.delta,
                "total_epochs": self.total_epochs,
                "seed": self.seed,
                "enabled": self.enabled,
                "num_samples": self.num_samples,
                "anneal_start_epoch_1indexed": self.anneal_start_epoch + 1,
                "last_epoch_stats": self.last_epoch_stats,
            }, f, indent=2)

    def load(self, ckpt_dir):
        """Restore scores. Returns True if state was found."""
        spath = os.path.join(ckpt_dir, "infobatch_scores.npy")
        if not os.path.exists(spath):
            logger.warning(
                f"No infobatch_scores.npy in {ckpt_dir}; starting from scores of 1.0."
            )
            return False
        scores = np.load(spath)
        if scores.shape[0] != self.num_samples:
            raise ValueError(
                f"infobatch_scores.npy has {scores.shape[0]} entries but the "
                f"dataset has {self.num_samples}. Refusing to resume."
            )
        self.scores = scores.astype(np.float32)
        epath = os.path.join(ckpt_dir, "infobatch_ever_scored.npy")
        if os.path.exists(epath):
            self.ever_scored = np.load(epath)
        logger.info(
            f"Restored InfoBatch scores: mean={self.scores.mean():.4f}, "
            f"{int(self.ever_scored.sum()):,} sequences scored at least once"
        )
        return True



def weighted_mlm_loss(logits, labels, sample_weights):
    """MLM cross-entropy with a weight per sequence.

    logits is the (batch, length, vocab) tensor from BertForMaskedLM. The model
    computes predictions only at masked positions and scatters them back with a
    differentiable index_put, so gradients flow through this loss even though
    the other positions hold zeros.

    loss = sum over masked tokens of (weight of the token's sequence * token CE)
           / number of masked tokens
    With all weights equal to 1 this is the model's own MLM loss.

    Returns (loss, per_seq_loss_sum, per_seq_loss_count).
    """
    mask = labels != -100
    n_masked = int(mask.sum())
    if n_masked == 0:
        zero = logits.sum() * 0.0
        b = labels.shape[0]
        return (zero,
                torch.zeros(b, device=labels.device),
                torch.zeros(b, device=labels.device))

    seq_idx = mask.nonzero(as_tuple=True)[0]        # (M,) sequence of each masked token
    flat_logits = logits[mask]                       # (M, V)
    flat_labels = labels[mask]                       # (M,)

    tok_loss = torch.nn.functional.cross_entropy(
        flat_logits.float(), flat_labels, reduction="none"
    )                                                # (M,)

    w = sample_weights[seq_idx]                      # (M,)
    loss = (tok_loss * w).sum() / float(n_masked)

    with torch.no_grad():
        b = labels.shape[0]
        det = tok_loss.detach()
        loss_sum = torch.zeros(b, device=labels.device, dtype=torch.float32)
        loss_cnt = torch.zeros(b, device=labels.device, dtype=torch.float32)
        loss_sum.index_add_(0, seq_idx, det)
        loss_cnt.index_add_(0, seq_idx, torch.ones_like(det))

    return loss, loss_sum, loss_cnt
