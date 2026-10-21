#!/usr/bin/env python
"""
NT-500M MLM pretraining on NT-HRG with B-PAS (batch pruning by activation
stability, Alam et al., 2026), used as a dynamic-pruning baseline.

Before training, the corpus is split once into fixed global batches of
per_device_train_batch_size * world_size sequences, using the seed. A sequence
stays in the same batch for the whole run; only the order of the batches and
the order inside each batch are shuffled every epoch. The few sequences that do
not fill a complete global batch are not used.

Forward hooks on the ESM encoder layers record the standard deviation of each
layer's output, averaged over the layers for every batch. From the second epoch
on, a batch whose average changed by at most delta(i) = delta_s * exp(alpha * i)
since the previous epoch is removed for the rest of the run, where
alpha = ln(delta_e / delta_s) / I and I is the number of epochs that fit in
max_steps. With "bpas_enabled": false nothing is pruned.

Sequences are numbered by their position among the non-empty lines of
train_file, as in train_nt.py. The run saves batch_groups.npy, the line_ids in
each batch, and pruned_sequence_mask.npy, a per-sequence mask indexed by
line_id, so B-PAS runs can be compared sequence by sequence with the other
methods. pruned_mask.npy is the per-batch mask.

Everything else (model, tokenizer, masking, optimizer, learning-rate schedule,
random offsets) is the same as in train_nt.py.

Launch with one process per GPU:
    torchrun --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> train_nt_bpas.py <config.json>
"""

import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import EsmConfig, EsmForMaskedLM, set_seed

# The NT tokenizer lives in ../tokenizer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tokenizer"))
from nt_tokenizer import NTTokenizer  # noqa: E402

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train_nt")


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    # Defaults
    cfg.setdefault("gradient_accumulation_steps", 1)
    cfg.setdefault("max_grad_norm", 1.0)
    cfg.setdefault("dataloader_num_workers", 4)
    cfg.setdefault("save_steps", 5000)
    cfg.setdefault("eval_steps", 5000)
    cfg.setdefault("save_total_limit", 50)
    cfg.setdefault("logging_steps", 100)
    cfg.setdefault("seed", 42)
    cfg.setdefault("bf16", True)
    cfg.setdefault("mlm_probability", 0.15)
    cfg.setdefault("random_replace_fraction", 0.10)
    cfg.setdefault("max_tokens_per_sequence", 1000)
    cfg.setdefault("overwrite_output_dir", False)
    # B-PAS thresholds (delta_s, delta_e) and an optional early stop when few batches remain
    cfg.setdefault("bpas_enabled", True)
    cfg.setdefault("bpas_delta_start", 1e-4)
    cfg.setdefault("bpas_delta_end", 1e-3)
    cfg.setdefault("bpas_min_batches_fraction", 0.0)
    # Stop after this many epochs (counted from 1, like "Epoch N" in the logs);
    # None runs until max_steps
    cfg.setdefault("graceful_exit_epoch", None)
    return cfg


class NTPretrainDataset(Dataset):
    """One 6,100 bp window per non-empty line of the text file.

    A sequence's index in the dataset (its line_id) is its position among the
    non-empty lines, counting from 0, so it is the same in every run on that file.
    Each epoch, every sequence starts at a random offset between 0 and 100 bp
    (the augmentation of the NT paper), and the next 6,000 bp are tokenized
    into at most 1,000 6-mer tokens. The offset depends only on the epoch and
    the sequence, so it is reproducible.
    """

    def __init__(self, file_path: str, tokenizer: NTTokenizer, max_tokens: int = 1000):
        logger.info(f"Loading sequences from {file_path} ...")
        with open(file_path) as f:
            self.sequences = [line.strip() for line in f if line.strip()]
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.epoch = 0
        logger.info(f"  Loaded {len(self.sequences):,} sequences")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Random offset augmentation (deterministic per epoch+idx for reproducibility)
        rng = np.random.RandomState(self.epoch * len(self.sequences) + idx)
        max_offset = max(0, len(seq) - 6000)
        offset = rng.randint(0, min(max_offset, 100) + 1) if max_offset > 0 else 0
        sub = seq[offset:]
        token_ids = self.tokenizer.tokenize_sequence(sub, max_tokens=self.max_tokens)
        return token_ids


def collate_fn(batch: List[List[int]], pad_id: int = 1, cls_id: int = 3):
    """Prepend <cls> and pad to the longest sequence in the batch."""
    max_len = max(len(ids) for ids in batch) + 1  # +1 for CLS
    input_ids = []
    attention_mask = []
    for ids in batch:
        seq = [cls_id] + ids
        pad_len = max_len - len(seq)
        attention_mask.append([1] * len(seq) + [0] * pad_len)
        seq += [pad_id] * pad_len
        input_ids.append(seq)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def mask_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mlm_prob: float = 0.15,
    mask_id: int = 2,
    pad_id: int = 1,
    cls_id: int = 3,
    vocab_size: int = 4105,
    random_replace_frac: float = 0.10,
):
    """BERT-style 80/10/10 masking of a batch.

    Returns (masked_input_ids, labels), with labels = -100 at positions that are not masked.
    """
    labels = input_ids.clone()
    prob_matrix = torch.full(input_ids.shape, mlm_prob, device=input_ids.device)

    # Never mask special / pad
    prob_matrix[input_ids == cls_id] = 0.0
    prob_matrix[input_ids == pad_id] = 0.0
    prob_matrix[attention_mask == 0] = 0.0

    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100

    # 80% -> [MASK]
    replace_mask = torch.bernoulli(
        torch.full(input_ids.shape, 0.8, device=input_ids.device)
    ).bool() & masked_indices
    input_ids[replace_mask] = mask_id

    # 10% -> random token (never one of the 4 special tokens)
    random_mask = (
        torch.bernoulli(
            torch.full(input_ids.shape, 0.5, device=input_ids.device)
        ).bool()
        & masked_indices
        & ~replace_mask
    )
    random_tokens = torch.randint(4, vocab_size, input_ids.shape, device=input_ids.device)
    input_ids[random_mask] = random_tokens[random_mask]

    # remaining 10% -> keep the original token
    return input_ids, labels


def get_nt_lr_lambda(warmup_steps: int, peak_lr: float, start_lr: float):
    """LambdaLR factor for the NT schedule: linear warmup from start_lr to peak_lr, then
    inverse square root decay, never below 1% of peak_lr."""
    ratio = start_lr / peak_lr

    def lr_lambda(step):
        if step < warmup_steps:
            return ratio + (1.0 - ratio) * step / warmup_steps
        return max((warmup_steps / max(step, 1)) ** 0.5, 0.01)

    return lr_lambda


class BPASBatchSampler:
    """Yields this rank's share of each fixed global batch.

    Each global batch keeps the same sequences for the whole run, which is what
    makes the epoch-to-epoch activation comparison meaningful. Every epoch the
    sampler skips pruned batches, shuffles the order of the active batches (the
    same order on all ranks), and for each batch yields this rank's
    local_batch_size indices in shuffled order. It is passed to the DataLoader
    as batch_sampler.
    """

    def __init__(self, batch_groups, world_size, rank, local_batch_size, seed):
        self.batch_groups = batch_groups
        self.world_size = world_size
        self.rank = rank
        self.local_batch_size = local_batch_size
        # Advanced once per epoch, identically on all ranks
        self._epoch_rng = np.random.default_rng(seed)
        self._all_batch_ids = np.arange(len(batch_groups), dtype=np.int64)
        self._active_ids = self._all_batch_ids.copy()
        self._yielded_order: List[int] = []

    def set_active_ids(self, active_ids: np.ndarray):
        self._active_ids = np.array(sorted(active_ids), dtype=np.int64)

    def __iter__(self):
        # Seed for this epoch (the same on all ranks)
        epoch_seed = int(self._epoch_rng.integers(0, 2**31))
        rng = np.random.default_rng(epoch_seed)

        shuffled = rng.permutation(self._active_ids)
        self._yielded_order = shuffled.tolist()

        r, bs = self.rank, self.local_batch_size
        for bid in shuffled:
            positions = self.batch_groups[int(bid)]
            local = positions[r * bs : (r + 1) * bs].copy()
            rng.shuffle(local)
            yield local.tolist()

    def __len__(self):
        return len(self._active_ids)

    @property
    def yielded_batch_ids(self) -> List[int]:
        """Order of batch_ids yielded in the most recent __iter__ call."""
        return self._yielded_order

    @property
    def num_batches_total(self) -> int:
        return len(self._all_batch_ids)


def bpas_threshold(delta_s: float, delta_e: float,
                   total_epochs: int, epoch: int) -> float:
    """Pruning threshold for an epoch (Alam et al., 2026):
        delta(i) = delta_s * exp(alpha * i),   alpha = ln(delta_e / delta_s) / I
    """
    if total_epochs <= 0:
        return delta_s
    alpha = math.log(delta_e / delta_s) / total_epochs
    exponent = alpha * epoch
    if exponent > 700:  # avoid overflow in math.exp
        return delta_e
    return min(delta_s * math.exp(exponent), delta_e)


def install_activation_hooks(model_module, encoder_attr_holder):
    """Register a forward hook on each ESM encoder layer that records the std of its output.

    The stds are kept as GPU tensors in encoder._layer_stds and averaged once per
    forward pass, so the hooks do not force a GPU-to-CPU copy for every layer.
    Returns the hook handles so they can be removed at the end.
    """
    encoder = model_module.esm.encoder
    encoder._layer_stds = []
    encoder_attr_holder["encoder"] = encoder

    def _hook(module, inp, out):
        # EsmLayer returns a tuple whose first element is the hidden states (batch, length, dim)
        h = out[0] if isinstance(out, tuple) else out
        with torch.no_grad():

            std_t = h.detach().float().std()
        encoder._layer_stds.append(std_t)

    handles = []
    for layer in encoder.layer:
        handles.append(layer.register_forward_hook(_hook))
    return handles


def save_sequence_mask(path, batch_groups, pruned_mask, num_train_sequences):
    """Save a per-sequence mask indexed by line_id, like pruned_mask.npy in train_nt.py.

    True marks a sequence that is no longer trained: it is in a pruned batch, or
    it is one of the leftover sequences that do not fill a complete global batch.
    """
    seq_mask = np.ones(num_train_sequences, dtype=bool)
    seq_mask[batch_groups[~pruned_mask].ravel()] = False
    np.save(path, seq_mask)


def save_checkpoint(
    model, optimizer, scheduler, tokenizer,
    global_step, current_epoch, display_epoch,
    pruned_mask, prev_std_dict,
    num_batches_total, num_train_sequences, global_batch_size,
    max_steps, log_history,
    output_dir, save_total_limit, rank,
    train_start_time, world_size, cfg,
):
    """Save the model, optimizer and scheduler state, B-PAS state, and a run summary (rank 0 only)."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")

    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

        # Model checkpoint in Hugging Face format (loads with from_pretrained)
        model.module.save_pretrained(ckpt_dir, safe_serialization=False)
        tokenizer.save_pretrained(ckpt_dir)

        # Optimizer and scheduler state, for resuming
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": current_epoch,
                "current_epoch": current_epoch,
            },
            os.path.join(ckpt_dir, "training_state.pt"),
        )

        # B-PAS state: per-batch pruning mask and last epoch's stds
        np.save(os.path.join(ckpt_dir, "pruned_mask.npy"), pruned_mask)
        with open(os.path.join(ckpt_dir, "bpas_state.json"), "w") as f:
            json.dump({str(k): v for k, v in prev_std_dict.items()}, f)

        # trainer_state.json: human-readable run summary
        total_pruned = int(pruned_mask.sum())
        active_batches = num_batches_total - total_pruned
        elapsed_hours = (time.time() - train_start_time) / 3600.0
        trainer_state = {
            "global_step": global_step,
            "epoch": round(display_epoch, 6),
            "epoch_integer": current_epoch + 1,  # counted from 1
            "max_steps": max_steps,
            "learning_rate": scheduler.get_last_lr()[0],
            "total_sequences": num_train_sequences,
            "global_batch_size": global_batch_size,
            "total_batches": num_batches_total,
            "active_batches": active_batches,
            "active_sequences": active_batches * global_batch_size,
            "pruned_batches": total_pruned,
            "pruned_pct": round(100.0 * total_pruned / max(1, num_batches_total), 4),
            "bpas_enabled": cfg.get("bpas_enabled", True),
            "bpas_delta_start": cfg["bpas_delta_start"],
            "bpas_delta_end": cfg["bpas_delta_end"],
            "elapsed_hours": round(elapsed_hours, 2),
            "gpu_hours": round(elapsed_hours * world_size, 2),
            "log_history": log_history,
        }
        with open(os.path.join(ckpt_dir, "trainer_state.json"), "w") as f:
            json.dump(trainer_state, f, indent=2)
        with open(os.path.join(output_dir, "trainer_state.json"), "w") as f:
            json.dump(trainer_state, f, indent=2)

        logger.info(f"Saved checkpoint: {ckpt_dir}")

        # Rotate old checkpoints
        if save_total_limit and save_total_limit > 0:
            ckpts = sorted(
                [
                    d
                    for d in os.listdir(output_dir)
                    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
                ],
                key=lambda x: int(x.split("-")[1]),
            )
            while len(ckpts) > save_total_limit:
                old = os.path.join(output_dir, ckpts.pop(0))
                shutil.rmtree(old, ignore_errors=True)
                logger.info(f"Deleted old checkpoint: {old}")

    if dist.is_initialized():
        dist.barrier()


def find_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    ckpts = [
        d
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not ckpts:
        return None
    latest = max(ckpts, key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, latest)


def load_bpas_state(ckpt_dir, num_complete_batches):
    """Load the per-batch pruning mask, the last stds and the log history from a checkpoint folder."""
    pruned_mask = None
    prev_std_dict: Dict[int, float] = {}
    log_history = []

    pm_path = os.path.join(ckpt_dir, "pruned_mask.npy")
    if os.path.exists(pm_path):
        m = np.load(pm_path)
        if len(m) == num_complete_batches:
            pruned_mask = m
        else:
            logger.warning(
                f"pruned_mask.npy length {len(m)} != current num_complete_batches "
                f"{num_complete_batches}; ignoring."
            )

    bpas_path = os.path.join(ckpt_dir, "bpas_state.json")
    if os.path.exists(bpas_path):
        with open(bpas_path) as f:
            raw = json.load(f)
            prev_std_dict = {int(k): float(v) for k, v in raw.items()}

    ts_path = os.path.join(ckpt_dir, "trainer_state.json")
    if os.path.exists(ts_path):
        with open(ts_path) as f:
            ts_json = json.load(f)
            log_history = ts_json.get("log_history", [])

    return pruned_mask, prev_std_dict, log_history


def run_epoch_end_bpas(
    epoch_std_buffer, batch_acc_buffer, prev_std_dict, pruned_mask,
    num_batches_total, current_epoch, total_epochs_est,
    delta_start, delta_end, min_batches_fraction,
    global_batch_size, local_batch_size,
    tracking_dir, device, train_start_time,
):
    """End-of-epoch B-PAS step (the same as in dnabert2/pretraining/run_mlm_bpas.py).

      1. Average each batch's activation std across ranks.
      2. Gather each batch's per-sequence MLM accuracy from all ranks (for the report only).
      3. From the second epoch on, prune batches whose std changed by at most the threshold.
      4. Rank 0 writes the per-batch report and the state needed to resume.
      5. Send the updated per-batch mask to all ranks.

    Returns (pruned_mask, current_std_dict).
    """
    is_dist = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_dist else 1
    rank = dist.get_rank() if is_dist else 0

    # Step 1: average the stds across ranks
    local_bids = sorted(epoch_std_buffer.keys())
    n_local = len(local_bids)

    if n_local > 0:
        stds_tensor = torch.tensor(
            [epoch_std_buffer[b] for b in local_bids],
            dtype=torch.float32, device=device,
        )
        if is_dist:
            dist.all_reduce(stds_tensor, op=dist.ReduceOp.SUM)
            stds_tensor /= world_size
        current_std_dict: Dict[int, float] = {
            b: s for b, s in zip(local_bids, stds_tensor.tolist())
        }
    else:
        current_std_dict = {}

    # Step 2: gather per-sequence accuracy from all ranks
    # Each rank holds local_batch_size accuracies per batch; gathering them
    # gives all global_batch_size accuracies of the batch
    global_batch_acc: Dict[int, np.ndarray] = {}
    if n_local > 0:
        local_acc_tensor = torch.zeros(
            n_local, local_batch_size, dtype=torch.float32, device=device,
        )
        for i, bid in enumerate(local_bids):
            if bid in batch_acc_buffer:
                arr = batch_acc_buffer[bid]
                local_acc_tensor[i, :len(arr)] = torch.from_numpy(arr).to(device)

        if is_dist:
            gathered = [torch.zeros_like(local_acc_tensor) for _ in range(world_size)]
            dist.all_gather(gathered, local_acc_tensor)
            full_acc = torch.cat(gathered, dim=1).cpu().numpy()
        else:
            full_acc = local_acc_tensor.cpu().numpy()

        for i, bid in enumerate(local_bids):
            global_batch_acc[bid] = full_acc[i]
    elif is_dist:
        # This rank processed no batches but must still join the all_gather
        dummy = torch.zeros(1, local_batch_size, dtype=torch.float32, device=device)
        gathered = [torch.zeros_like(dummy) for _ in range(world_size)]
        dist.all_gather(gathered, dummy)

    # Step 3: threshold and pruning
    threshold = bpas_threshold(delta_start, delta_end, total_epochs_est, current_epoch)

    newly_pruned: List[int] = []
    if current_epoch >= 1 and prev_std_dict and current_std_dict:
        already_pruned = set(np.where(pruned_mask)[0].tolist())
        common_bids = (set(current_std_dict.keys()) &
                       set(prev_std_dict.keys()) - already_pruned)
        common_arr = np.array(sorted(common_bids), dtype=np.int64)
        if len(common_arr):
            curr_vals = np.array([current_std_dict[b] for b in common_arr])
            prev_vals = np.array([prev_std_dict[b] for b in common_arr])
            stable_mask = np.abs(curr_vals - prev_vals) <= threshold
            newly_pruned = common_arr[stable_mask].tolist()
            for bid in newly_pruned:
                pruned_mask[bid] = True

    total_pruned = int(pruned_mask.sum())
    active_batches = num_batches_total - total_pruned

    # Step 4: rank 0 writes the report
    if rank == 0:
        os.makedirs(tracking_dir, exist_ok=True)
        elapsed_h = (time.time() - train_start_time) / 3600.0

        stats_path = os.path.join(
            tracking_dir, f"epoch_{current_epoch + 1:04d}_bpas_stats.txt")
        with open(stats_path, "w") as f:
            f.write(
                f"B-PAS Epoch {current_epoch + 1} | "
                f"threshold={threshold:.4e} | "
                f"newly_pruned={len(newly_pruned)} batches | "
                f"total_pruned={total_pruned}/{num_batches_total} | "
                f"active={active_batches} batches "
                f"({active_batches * global_batch_size:,} seqs) | "
                f"GPU-hours={elapsed_h * world_size:.2f}\n\n"
            )
            f.write(
                f"{'batch_id':>10}  {'mean_std':>12}  {'prev_std':>12}  "
                f"{'|delta|':>12}  {'action':>14}  "
                f"{'acc_mean':>10}  {'acc_median':>10}  "
                f"{'acc_min':>10}  {'acc_max':>10}  "
                f"{'pct_100':>10}\n"
            )
            f.write("-" * 125 + "\n")
            for bid in sorted(current_std_dict):
                curr = current_std_dict[bid]
                prev = prev_std_dict.get(bid, float("nan"))
                dv = abs(curr - prev) if not math.isnan(prev) else float("nan")
                if bid in newly_pruned:
                    action = "PRUNED"
                elif pruned_mask[bid]:
                    action = "already-pruned"
                else:
                    action = "kept"

                # Accuracy statistics over all sequences in the global batch
                if bid in global_batch_acc:
                    acc_arr = global_batch_acc[bid]
                    valid = acc_arr[acc_arr >= 0]  # leave out -1 (no [MASK] positions)
                    if len(valid) > 0:
                        a_mean = float(np.mean(valid))
                        a_median = float(np.median(valid))
                        a_min = float(np.min(valid))
                        a_max = float(np.max(valid))
                        pct_100 = float(np.sum(valid == 1.0) / len(valid) * 100)
                    else:
                        a_mean = a_median = a_min = a_max = pct_100 = float("nan")
                else:
                    a_mean = a_median = a_min = a_max = pct_100 = float("nan")

                f.write(
                    f"{bid:>10}  {curr:>12.6f}  {prev:>12.6f}  "
                    f"{dv:>12.6f}  {action:>14}  "
                    f"{a_mean:>10.4f}  {a_median:>10.4f}  "
                    f"{a_min:>10.4f}  {a_max:>10.4f}  "
                    f"{pct_100:>9.2f}%\n"
                )

        logger.info(
            f"B-PAS epoch {current_epoch + 1}: threshold={threshold:.4e} | "
            f"newly_pruned={len(newly_pruned)} | total_pruned={total_pruned} | "
            f"active_batches={active_batches} "
            f"({active_batches * global_batch_size:,} seqs) | "
            f"GPU-hours={elapsed_h * world_size:.2f}"
        )

        # Save the per-batch mask and this epoch's stds (used when resuming)
        np.save(os.path.join(tracking_dir, "pruned_mask.npy"), pruned_mask)
        with open(os.path.join(tracking_dir, "bpas_prev_std_dict.json"), "w") as f:
            json.dump({str(k): v for k, v in current_std_dict.items()}, f)

    # Step 5: send the mask to all ranks
    if is_dist:
        pm_tensor = torch.from_numpy(pruned_mask.view(np.uint8)).to(device)
        dist.broadcast(pm_tensor, src=0)
        pruned_mask = pm_tensor.cpu().numpy().view(bool).copy()
        dist.barrier()

    return pruned_mask, current_std_dict


@torch.no_grad()
def evaluate(model, dataloader, cfg, device):
    model.eval()
    total_loss, total_correct, total_masked = 0.0, 0, 0
    max_batches = 200  # evaluate on at most 200 batches per rank

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        masked_ids, labels = mask_tokens(
            input_ids.clone(),
            attention_mask,
            mlm_prob=cfg["mlm_probability"],
            vocab_size=cfg["vocab_size"],
            random_replace_frac=cfg.get("random_replace_fraction", 0.10),
        )

        outputs = model(
            input_ids=masked_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()

        preds = outputs.logits.argmax(dim=-1)
        mask = labels != -100
        total_correct += (preds[mask] == labels[mask]).sum().item()
        total_masked += mask.sum().item()

    # All-reduce across ranks
    stats = torch.tensor(
        [total_loss, total_correct, total_masked, i + 1],
        device=device, dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(stats)

    avg_loss = stats[0].item() / max(stats[3].item(), 1)
    accuracy = stats[1].item() / max(stats[2].item(), 1)
    model.train()
    return {"eval_loss": avg_loss, "eval_accuracy": accuracy}


def main():
    # config
    config_path = sys.argv[1]
    cfg = load_config(config_path)

    # DDP
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        logger.info(f"World size: {world_size}")
        logger.info(f"Config: {json.dumps(cfg, indent=2)}")

    set_seed(cfg.get("seed", 42))

    # tokenizer
    tokenizer = NTTokenizer(
        vocab_file=cfg["vocab_file"],
        model_max_length=cfg.get("max_tokens_per_sequence", 1000),
    )

    # output dirs
    output_dir = cfg["output_dir"]
    tracking_dir = os.path.join(output_dir, "bpas_tracking")
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        os.makedirs(tracking_dir, exist_ok=True)

    # model (resume or fresh)
    latest_ckpt = (
        find_latest_checkpoint(output_dir)
        if not cfg.get("overwrite_output_dir", False)
        else None
    )

    if latest_ckpt:
        if rank == 0:
            logger.info(f"Resuming from {latest_ckpt}")
        model = EsmForMaskedLM.from_pretrained(latest_ckpt)
    else:
        if rank == 0:
            logger.info("Initialising model from scratch")
        esm_config = EsmConfig(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            intermediate_size=cfg["intermediate_size"],
            max_position_embeddings=cfg["max_position_embeddings"],
            hidden_dropout_prob=cfg.get("hidden_dropout_prob", 0.0),
            attention_probs_dropout_prob=cfg.get("attention_probs_dropout_prob", 0.0),
            pad_token_id=tokenizer.pad_token_id,
            mask_token_id=tokenizer.mask_token_id,
            position_embedding_type=cfg.get("position_embedding_type", "absolute"),
            token_dropout=cfg.get("token_dropout", True),
            emb_layer_norm_before=False,
        )
        model = EsmForMaskedLM(esm_config)

    # Remove the unused contact prediction head (for a new model and when resuming)
    if hasattr(model.esm, "contact_head"):
        del model.esm.contact_head

    model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Model parameters: {n_params:.1f}M")

    # Activation std hooks on the encoder layers
    encoder_holder = {}
    hook_handles = install_activation_hooks(model.module, encoder_holder)
    encoder = encoder_holder["encoder"]
    if rank == 0:
        logger.info(
            f"B-PAS: activation std hooks installed on {len(encoder.layer)} encoder layers"
        )

    # datasets
    train_dataset = NTPretrainDataset(
        cfg["train_file"], tokenizer, max_tokens=cfg.get("max_tokens_per_sequence", 1000)
    )
    eval_dataset = NTPretrainDataset(
        cfg["validation_file"], tokenizer, max_tokens=cfg.get("max_tokens_per_sequence", 1000)
    )
    num_train_sequences = len(train_dataset)
    if rank == 0:
        logger.info(f"Training sequences: {num_train_sequences:,}")
        logger.info(f"Eval sequences: {len(eval_dataset):,}")

    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    _collate = lambda batch: collate_fn(batch, pad_id=pad_id, cls_id=cls_id)

    # optimizer
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.get("weight_decay", 0.0),
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.Adam(
        params,
        lr=cfg["peak_lr"],
        betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
        eps=cfg["adam_epsilon"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        get_nt_lr_lambda(cfg["warmup_steps"], cfg["peak_lr"], cfg["start_lr"]),
    )

    global_batch_size = cfg["per_device_train_batch_size"] * world_size
    local_batch_size = cfg["per_device_train_batch_size"]

    # Fixed global batches, assigned once from the seed (the same on all ranks)
    rng = np.random.default_rng(cfg.get("seed", 42))
    shuffled_idx = rng.permutation(num_train_sequences).astype(np.int64)

    num_complete_batches = num_train_sequences // global_batch_size
    num_used = num_complete_batches * global_batch_size
    num_tail = num_train_sequences - num_used

    # batch_groups[bid] holds the global_batch_size line_ids of batch bid
    batch_groups = shuffled_idx[:num_used].reshape(num_complete_batches, global_batch_size)

    if rank == 0:
        logger.info(
            f"B-PAS: {num_train_sequences:,} sequences -> "
            f"{num_complete_batches:,} fixed global batches of "
            f"{global_batch_size} (per_device={local_batch_size} x "
            f"world_size={world_size})"
        )
        if num_tail > 0:
            logger.info(
                f"B-PAS: {num_tail:,} tail sequences dropped "
                f"({100.0 * num_tail / num_train_sequences:.4f}%)"
            )
        # Which line_ids are in which batch; together with the per-batch mask this
        # gives the per-sequence picture
        np.save(os.path.join(tracking_dir, "batch_groups.npy"), batch_groups.astype(np.int32))

    batch_sampler = BPASBatchSampler(
        batch_groups=batch_groups,
        world_size=world_size,
        rank=rank,
        local_batch_size=local_batch_size,
        seed=cfg.get("seed", 42),
    )

    # Number of epochs I in the threshold schedule, estimated from max_steps
    grad_accum = cfg["gradient_accumulation_steps"]
    max_steps = cfg["max_steps"]
    steps_per_epoch_initial = num_complete_batches // grad_accum
    total_epochs_est = (
        math.ceil(max_steps / steps_per_epoch_initial)
        if steps_per_epoch_initial > 0 else 1
    )
    if rank == 0:
        logger.info(
            f"B-PAS: total_epochs_est={total_epochs_est} "
            f"(steps_per_epoch_initial={steps_per_epoch_initial}, "
            f"max_steps={max_steps}). Fixed for the whole run."
        )
        logger.info(
            f"B-PAS: bpas_enabled={cfg['bpas_enabled']}, "
            f"delta_start={cfg['bpas_delta_start']:.4e}, "
            f"delta_end={cfg['bpas_delta_end']:.4e}, "
            f"min_batches_fraction={cfg['bpas_min_batches_fraction']}, "
            f"graceful_exit_epoch={cfg['graceful_exit_epoch']}"
        )

    # pruning state + resume
    pruned_mask = np.zeros(num_complete_batches, dtype=bool)
    prev_std_dict: Dict[int, float] = {}
    global_step = 0
    current_epoch = 0
    log_history = []

    if latest_ckpt:
        state_path = os.path.join(latest_ckpt, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            global_step = state["global_step"]
            current_epoch = state.get("current_epoch", state.get("epoch", 0))
            if rank == 0:
                logger.info(f"Resumed optimizer/scheduler at step {global_step}, epoch {current_epoch}")

        loaded_mask, loaded_std, loaded_history = load_bpas_state(latest_ckpt, num_complete_batches)
        if loaded_mask is not None:
            pruned_mask = loaded_mask
        prev_std_dict = loaded_std
        log_history = loaded_history
        if rank == 0:
            logger.info(
                f"Resumed B-PAS state: pruned {int(pruned_mask.sum()):,}/{num_complete_batches:,} "
                f"batches, prev_std_dict has {len(prev_std_dict):,} entries, "
                f"log_history has {len(log_history)} entries"
            )
    elif os.path.exists(os.path.join(tracking_dir, "pruned_mask.npy")):
        # No full checkpoint but a pruning mask exists in the tracking folder (edge case)
        m = np.load(os.path.join(tracking_dir, "pruned_mask.npy"))
        if len(m) == num_complete_batches:
            pruned_mask = m
            if rank == 0:
                logger.info(f"Loaded existing pruned_mask: {int(pruned_mask.sum()):,} batches pruned")
        bpas_std_path = os.path.join(tracking_dir, "bpas_prev_std_dict.json")
        if os.path.exists(bpas_std_path):
            with open(bpas_std_path) as f:
                raw = json.load(f)
                prev_std_dict = {int(k): float(v) for k, v in raw.items()}

    # Send rank 0's pruned_mask to all ranks
    pm_tensor = torch.from_numpy(pruned_mask.view(np.uint8)).to(device)
    if dist.is_initialized():
        dist.broadcast(pm_tensor, src=0)
    pruned_mask = pm_tensor.cpu().numpy().view(bool).copy()

    # Update sampler with current active set
    active_ids = np.where(~pruned_mask)[0]
    batch_sampler.set_active_ids(active_ids)

    # training config
    use_bf16 = cfg.get("bf16", True)
    save_steps = cfg["save_steps"]
    eval_steps = cfg["eval_steps"]
    logging_steps = cfg.get("logging_steps", 100)
    save_total_limit = cfg.get("save_total_limit", 50)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

    if rank == 0:
        effective_bs = local_batch_size * world_size * grad_accum
        logger.info(
            f"Effective batch size: {local_batch_size} x {world_size} x {grad_accum} = "
            f"{effective_bs} sequences "
            f"({effective_bs * cfg.get('max_tokens_per_sequence', 1000) / 1e6:.2f}M tokens)"
        )
        logger.info(f"Max steps: {max_steps}, warmup: {cfg['warmup_steps']}, BF16: {use_bf16}")

    if rank == 0:
        logger.info("=" * 60)
        logger.info("TRAINING START")
        logger.info("=" * 60)

    model.train()
    train_start_time = time.time()
    accum_loss = 0.0

    while global_step < max_steps:
        # Active batches for this epoch
        active_ids = np.where(~pruned_mask)[0]
        batch_sampler.set_active_ids(active_ids)
        active_batch_count = len(active_ids)

        # The random offsets depend on the epoch
        train_dataset.set_epoch(current_epoch)

        # Build a fresh dataloader for this epoch
        epoch_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=_collate,
            num_workers=cfg.get("dataloader_num_workers", 4),
            pin_memory=True,
        )
        steps_this_epoch = len(epoch_loader) // grad_accum

        if rank == 0:
            logger.info(
                f"Epoch {current_epoch + 1}: {active_batch_count:,} active batches "
                f"({active_batch_count * global_batch_size:,} seqs), "
                f"{steps_this_epoch:,} steps/epoch "
                f"(pruned {int(pruned_mask.sum()):,}/{num_complete_batches:,})"
            )

        # Optional early stop when few batches remain (off when bpas_min_batches_fraction is 0)
        min_frac = cfg["bpas_min_batches_fraction"]
        if min_frac > 0 and active_batch_count < num_complete_batches * min_frac:
            if rank == 0:
                logger.info(
                    f"B-PAS early stop: only {active_batch_count} batches remain "
                    f"(< {min_frac*100:.1f}% of {num_complete_batches})"
                )
            break

        # Stop if every batch has been pruned
        if active_batch_count == 0:
            if rank == 0:
                logger.info(
                    f"B-PAS: all {num_complete_batches} batches pruned. "
                    f"Saving final checkpoint at step {global_step}."
                )
            display_epoch = float(current_epoch)
            save_checkpoint(
                model, optimizer, scheduler, tokenizer,
                global_step, current_epoch, display_epoch,
                pruned_mask, prev_std_dict,
                num_complete_batches, num_train_sequences, global_batch_size,
                max_steps, log_history,
                output_dir, save_total_limit, rank,
                train_start_time, world_size, cfg,
            )
            break

        # Per-epoch buffers
        epoch_std_buffer: Dict[int, float] = {}     # batch id -> mean activation std
        batch_acc_buffer: Dict[int, np.ndarray] = {}  # batch id -> this rank's per-sequence accuracies
        batch_ids_order: List[int] = []

        optimizer.zero_grad()
        accum_loss = 0.0
        micro_step = 0

        for batch_enum_idx, batch in enumerate(epoch_loader):
            if global_step >= max_steps:
                break

            # The sampler records this epoch's batch order once iteration starts
            if batch_enum_idx == 0:
                batch_ids_order = batch_sampler.yielded_batch_ids
            bid = batch_ids_order[batch_enum_idx]

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            masked_ids, labels = mask_tokens(
                input_ids.clone(),
                attention_mask,
                mlm_prob=cfg["mlm_probability"],
                vocab_size=cfg["vocab_size"],
                random_replace_frac=cfg.get("random_replace_fraction", 0.10),
            )

            # Start a fresh list of per-layer stds for this forward pass
            encoder._layer_stds.clear()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(
                    input_ids=masked_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / grad_accum

            loss.backward()
            accum_loss += loss.item()

            # Mean of the per-layer output stds for this batch
            if encoder._layer_stds:
                stacked = torch.stack(encoder._layer_stds)  # (num_layers,)
                mean_std = stacked.mean().item()
                epoch_std_buffer[bid] = mean_std

            # Per-sequence MLM accuracy within this batch (for the report)
            with torch.no_grad():
                preds = outputs.logits.argmax(dim=-1)
                mask_pos = labels != -100
                per_seq_correct = ((preds == labels) & mask_pos).sum(dim=-1).float()
                per_seq_total = mask_pos.sum(dim=-1).float()
                per_seq_acc = torch.where(
                    per_seq_total > 0,
                    per_seq_correct / per_seq_total,
                    torch.tensor(-1.0, device=device),
                )
                batch_acc_buffer[bid] = per_seq_acc.cpu().numpy()

            micro_step += 1

            # Optimizer step (every grad_accum micro-steps)
            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Fractional epoch for display
                frac = micro_step / max(1, len(epoch_loader))
                display_epoch = current_epoch + frac

                # Logging
                if rank == 0 and global_step % logging_steps == 0:
                    elapsed = time.time() - train_start_time
                    gpu_hours = world_size * elapsed / 3600
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"step={global_step}/{max_steps} | "
                        f"epoch={display_epoch:.2f} | "
                        f"loss={accum_loss:.4f} | lr={lr:.2e} | "
                        f"elapsed={elapsed/3600:.2f}h | gpu-hours={gpu_hours:.2f}"
                    )
                    log_history.append({
                        "step": global_step,
                        "epoch": round(display_epoch, 4),
                        "loss": round(accum_loss, 6),
                        "learning_rate": lr,
                        "active_batches": int(active_batch_count),
                        "pruned_batches": int(pruned_mask.sum()),
                    })
                accum_loss = 0.0

                # Eval
                if global_step % eval_steps == 0:
                    eval_sampler = DistributedSampler(
                        eval_dataset, num_replicas=world_size, rank=rank, shuffle=False,
                    )
                    eval_loader = DataLoader(
                        eval_dataset,
                        batch_size=cfg.get("per_device_eval_batch_size", local_batch_size),
                        sampler=eval_sampler,
                        collate_fn=_collate,
                        num_workers=2,
                        pin_memory=True,
                    )
                    metrics = evaluate(model, eval_loader, cfg, device)
                    if rank == 0:
                        gpu_hours = world_size * (time.time() - train_start_time) / 3600
                        logger.info(
                            f"[Eval step {global_step}] loss={metrics['eval_loss']:.4f} "
                            f"acc={metrics['eval_accuracy']:.4f} | gpu-hours={gpu_hours:.2f}"
                        )
                        log_history.append({
                            "step": global_step,
                            "epoch": round(display_epoch, 4),
                            "eval_loss": round(metrics["eval_loss"], 6),
                            "eval_accuracy": round(metrics["eval_accuracy"], 6),
                        })

                # Save
                if global_step % save_steps == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, tokenizer,
                        global_step, current_epoch, display_epoch,
                        pruned_mask, prev_std_dict,
                        num_complete_batches, num_train_sequences, global_batch_size,
                        max_steps, log_history,
                        output_dir, save_total_limit, rank,
                        train_start_time, world_size, cfg,
                    )

        # End of epoch: compare activation stds and prune stable batches
        if cfg["bpas_enabled"] and global_step < max_steps:
            pruned_mask, prev_std_dict = run_epoch_end_bpas(
                epoch_std_buffer, batch_acc_buffer, prev_std_dict, pruned_mask,
                num_complete_batches, current_epoch, total_epochs_est,
                cfg["bpas_delta_start"], cfg["bpas_delta_end"],
                cfg["bpas_min_batches_fraction"],
                global_batch_size, local_batch_size,
                tracking_dir, device, train_start_time,
            )
            if rank == 0:
                save_sequence_mask(os.path.join(tracking_dir, "pruned_sequence_mask.npy"),
                                   batch_groups, pruned_mask, num_train_sequences)
        current_epoch += 1

        if rank == 0:
            gpu_hours = world_size * (time.time() - train_start_time) / 3600
            logger.info(
                f"Epoch {current_epoch} complete at step {global_step} | "
                f"gpu-hours={gpu_hours:.2f}"
            )

        # Stop after the scheduled number of epochs (counted from 1). The learning-rate
        # schedule is still tied to max_steps; the run just ends earlier.
        graceful_exit_epoch = cfg.get("graceful_exit_epoch", None)
        if graceful_exit_epoch and current_epoch == graceful_exit_epoch:
            if rank == 0:
                exit_dir = os.path.join(
                    output_dir, f"epoch_{graceful_exit_epoch}_checkpoint")
                os.makedirs(exit_dir, exist_ok=True)
                model.module.save_pretrained(exit_dir, safe_serialization=False)
                tokenizer.save_pretrained(exit_dir)
                np.save(os.path.join(exit_dir, "pruned_mask.npy"), pruned_mask)
                with open(os.path.join(exit_dir, "bpas_state.json"), "w") as f:
                    json.dump({str(k): v for k, v in prev_std_dict.items()}, f)
                save_sequence_mask(os.path.join(exit_dir, "pruned_sequence_mask.npy"),
                                   batch_groups, pruned_mask, num_train_sequences)
                logger.info(
                    f"Epoch {graceful_exit_epoch} reached (step {global_step}) "
                    f"final model saved to {exit_dir}"
                )
            if dist.is_initialized():
                dist.barrier()
            break


    save_checkpoint(
        model, optimizer, scheduler, tokenizer,
        global_step, current_epoch, float(current_epoch),
        pruned_mask, prev_std_dict,
        num_complete_batches, num_train_sequences, global_batch_size,
        max_steps, log_history,
        output_dir, save_total_limit, rank,
        train_start_time, world_size, cfg,
    )

    if rank == 0:
        total_time = time.time() - train_start_time
        total_gpu_hours = world_size * total_time / 3600
        total_pruned = int(pruned_mask.sum())
        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE: {total_time / 3600:.2f}h wall-clock, "
                    f"{total_gpu_hours:.2f} GPU-hours")
        logger.info(
            f"B-PAS final: {total_pruned}/{num_complete_batches} batches pruned "
            f"({100.0*total_pruned/max(1,num_complete_batches):.2f}%)"
        )
        logger.info(f"  Steps : {global_step}")
        logger.info(f"  GPUs  : {world_size}")
        logger.info("=" * 60)

    # Remove the hooks
    for h in hook_handles:
        h.remove()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
