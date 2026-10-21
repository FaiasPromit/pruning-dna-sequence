#!/usr/bin/env python
"""
HyenaDNA CLM pretraining with B-PAS (batch pruning by activation stability,
Alam et al., 2026), used as a dynamic-pruning baseline.

The pruning works as in dnabert2/pretraining/run_mlm_bpas.py. Before training,
the corpus is split once into fixed global batches of
per_device_train_batch_size * world_size sequences, using the seed. A sequence
stays in the same batch for the whole run; only the order of the batches and the
order inside each batch are shuffled every epoch. The few sequences that do not
fill a complete global batch are not used.

For every batch, the standard deviation of the output of each Hyena block is
recorded and averaged over the blocks. From the second epoch on, a batch whose
average changed by at most delta(i) = delta_s * exp(alpha * i) since the
previous epoch is removed for the rest of the run, where
alpha = ln(delta_e / delta_s) / I and I is the number of full epochs that fit in
max_steps. Each sequence's next-token accuracy is recorded in the per-batch
report, but B-PAS does not use it.

Sequences are numbered by line_id exactly as in run_clm.py (their position among
the non-empty lines of train_file). The run saves batch_groups.npy, the line_ids
in each batch, and pruned_sequence_mask.npy, a per-sequence mask indexed by
line_id, so B-PAS runs can be compared sequence by sequence with the other
methods. pruned_mask.npy is the per-batch mask.

Everything else (model, optimizer, learning-rate schedule) is the same as in
run_clm.py.

Launch with one process per GPU:
    torchrun --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> run_clm_bpas.py <config.json>
"""

import json
import logging
import math
import os
import shutil
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset
from transformers import set_seed
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# The HyenaDNA model code and tokenizer live in ../model_code
MODEL_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_code")


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault("gradient_accumulation_steps", 1)
    cfg.setdefault("max_grad_norm", 1.0)
    cfg.setdefault("dataloader_num_workers", 4)
    cfg.setdefault("dataloader_pin_memory", True)
    cfg.setdefault("preprocessing_num_workers", 16)
    cfg.setdefault("save_steps", 5000)
    cfg.setdefault("eval_steps", 5000)
    cfg.setdefault("save_total_limit", 10)
    cfg.setdefault("logging_steps", 100)
    cfg.setdefault("seed", 42)
    cfg.setdefault("bf16", True)
    cfg.setdefault("max_seq_length", 1024)
    cfg.setdefault("lr_scheduler_type", "cosine")
    cfg.setdefault("warmup_steps", 0)               # learning-rate warmup (there is no sequence-length warmup)
    cfg.setdefault("weight_decay", 0.1)
    cfg.setdefault("adam_beta1", 0.9)
    cfg.setdefault("adam_beta2", 0.999)
    cfg.setdefault("adam_epsilon", 1e-8)
    cfg.setdefault("add_bos", False)                # by default the input is the nucleotides only
    cfg.setdefault("add_eos", False)
    # B-PAS defaults
    cfg.setdefault("bpas_delta_start", 1e-4)
    cfg.setdefault("bpas_delta_end", 1e-3)
    cfg.setdefault("bpas_min_batches_fraction", 0.0)
    # Stop after this many epochs (counted from 1, as in the "Epoch N" log lines);
    # null or 0 runs until max_steps
    cfg.setdefault("graceful_exit_epoch", None)
    return cfg


class CLMDataCollator:
    """Pads input_ids and builds the labels.

    The labels are a copy of input_ids with padding set to -100; the shift by one
    position happens inside HyenaDNAForCausalLM. All windows have the same length,
    so in practice nothing is padded.
    """

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features, return_tensors=None):
        seqs = [f["input_ids"] for f in features]
        maxlen = max(len(s) for s in seqs)
        bsz = len(seqs)

        input_ids = torch.full((bsz, maxlen), self.pad_token_id, dtype=torch.long)
        attn = torch.zeros((bsz, maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            n = len(s)
            input_ids[i, :n] = torch.as_tensor(s, dtype=torch.long)
            attn[i, :n] = 1

        labels = input_ids.clone()
        labels[attn == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def prepare_datasets(cfg, tokenizer, rank, world_size):
    """Load the text file (one window per line) and tokenize it.

    A sequence's index in train_dataset is its line_id, its position among the
    non-empty lines of train_file, the same numbering as in run_clm.py.
    Returns (train_dataset, eval_dataset, num_train_sequences).
    """
    data_files = {}
    if cfg.get("train_file"):
        data_files["train"] = cfg["train_file"]
    if cfg.get("validation_file"):
        data_files["validation"] = cfg["validation_file"]

    max_seq_length = cfg["max_seq_length"]
    add_bos = cfg["add_bos"]
    add_eos = cfg["add_eos"]
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    def tokenize_fn(examples):
        texts = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        enc = tokenizer(
            texts, add_special_tokens=False, padding=False,
            truncation=True, max_length=max_seq_length,
        )
        input_ids = enc["input_ids"]
        if add_bos or add_eos:
            pre = [bos_id] if add_bos else []
            post = [eos_id] if add_eos else []
            input_ids = [pre + ids + post for ids in input_ids]
        return {"input_ids": input_ids}

    num_proc = cfg["preprocessing_num_workers"]

    def _build():
        raw = load_dataset("text", data_files=data_files)
        return raw.map(tokenize_fn, batched=True, num_proc=num_proc,
                       remove_columns=["text"], load_from_cache_file=True,
                       desc="Tokenizing")

    # Rank 0 builds the cache first; the others read from it.
    if rank == 0:
        logger.info("Rank 0: loading and tokenizing dataset...")
        tokenized = _build()
        logger.info(f"Rank 0: dataset ready with {len(tokenized['train']):,} train sequences")
    if world_size > 1:
        dist.barrier()
    if rank != 0:
        tokenized = _build()

    train_dataset = tokenized["train"]
    eval_dataset = tokenized.get("validation", None)
    return train_dataset, eval_dataset, len(train_dataset)


def build_model(cfg, tokenizer, device):
    """Build HyenaDNAForCausalLM from the settings in the config file."""
    sys.path.insert(0, MODEL_CODE_DIR)
    from configuration_hyenadna import HyenaDNAConfig
    from modeling_hyenadna import HyenaDNAForCausalLM

    config = HyenaDNAConfig(
        vocab_size=cfg.get("vocab_size", tokenizer.vocab_size),
        d_model=cfg.get("d_model", 256),
        n_layer=cfg.get("n_layer", 8),
        d_inner=cfg.get("d_inner", 1024),
        max_seq_length=cfg["max_seq_length"],
        order=cfg.get("order", 2),
        filter_order=cfg.get("filter_order", 64),
        filter_emb_dim=cfg.get("filter_emb_dim", 5),
        filter_w=cfg.get("filter_w", 10.0),
        filter_lr=cfg.get("filter_lr", cfg.get("learning_rate", 6e-4)),
        filter_wd=cfg.get("filter_wd", 0.0),
        lr_pos_emb=cfg.get("lr_pos_emb", 1e-5),
        modulate=cfg.get("modulate", True),
        resid_dropout=cfg.get("resid_dropout", 0.0),
        embed_dropout=cfg.get("embed_dropout", 0.1),
        pad_vocab_size_multiple=cfg.get("pad_vocab_size_multiple", 1),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
    )

    model = HyenaDNAForCausalLM(config)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")
    return model.to(device), config


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
        """Update the set of active (non-pruned) batch IDs."""
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
    """
    Section 2.3 of the B-PAS paper:
        delta(i) = delta_s * exp(alpha * i)
        alpha    = (1/I) * ln(delta_e / delta_s)
    """
    if total_epochs <= 0:
        return delta_s
    alpha = math.log(delta_e / delta_s) / total_epochs
    exponent = alpha * epoch
    if exponent > 700:  # prevent math.exp overflow (float64 max ~ e^709)
        return delta_e
    return min(delta_s * math.exp(exponent), delta_e)


def save_sequence_mask(path, batch_groups, pruned_mask, num_train_sequences):
    """Save a per-sequence mask indexed by line_id, like pruned_mask.npy in run_clm.py.

    True marks a sequence that is no longer trained: it is in a pruned batch, or
    it is one of the leftover sequences that do not fill a complete global batch.
    """
    seq_mask = np.ones(num_train_sequences, dtype=bool)
    seq_mask[batch_groups[~pruned_mask].ravel()] = False
    np.save(path, seq_mask)


def make_train_dataloader(full_dataset, batch_sampler, data_collator, cfg):
    """Build a DataLoader driven by BPASBatchSampler.

    batch_sampler yields lists of dataset indices directly, so no batch_size,
    shuffle, sampler, or drop_last args are used.
    """
    return DataLoader(
        full_dataset,
        batch_sampler=batch_sampler,
        collate_fn=data_collator,
        num_workers=cfg["dataloader_num_workers"],
        pin_memory=cfg["dataloader_pin_memory"],
    )


def make_eval_dataloader(eval_dataset, data_collator, cfg, rank, world_size):
    """Build the evaluation DataLoader (the validation set is never pruned)."""
    sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank,
                                 shuffle=False, drop_last=False)
    return DataLoader(
        eval_dataset, batch_size=cfg.get("per_device_eval_batch_size", 64),
        sampler=sampler, collate_fn=data_collator,
        num_workers=cfg["dataloader_num_workers"], pin_memory=cfg["dataloader_pin_memory"],
    )


@torch.no_grad()
def evaluate(model, eval_dataloader, device, use_bf16):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
        total_loss += outputs.loss.item()
        num_batches += 1

        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        preds = shift_logits.argmax(dim=-1)
        valid = shift_labels != -100
        total_correct += ((preds == shift_labels) & valid).sum().item()
        total_tokens += valid.sum().item()

    stats = torch.tensor([total_loss, num_batches, total_correct, total_tokens],
                         dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    avg_loss = stats[0].item() / max(stats[1].item(), 1)
    accuracy = stats[2].item() / max(stats[3].item(), 1)
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")

    model.train()
    return {"eval_loss": avg_loss, "eval_accuracy": accuracy, "perplexity": perplexity}


def save_checkpoint(model, optimizer, scheduler, global_step, current_epoch,
                    display_epoch, pruned_mask, prev_std_dict,
                    num_batches_total, num_train_sequences,
                    global_batch_size, max_steps, log_history,
                    output_dir, save_total_limit, tokenizer, rank,
                    train_start_time, world_size, cfg):
    """Save the model, optimizer and scheduler state, B-PAS state, and a run summary (rank 0 only)."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")

    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

        # Model in Hugging Face format (loads with from_pretrained)
        model.module.save_pretrained(ckpt_dir, safe_serialization=False)
        tokenizer.save_pretrained(ckpt_dir)

        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "current_epoch": current_epoch,
        }, os.path.join(ckpt_dir, "training_state.pt"))

        # B-PAS state: the per-batch mask and the previous epoch's per-batch stds
        np.save(os.path.join(ckpt_dir, "pruned_mask.npy"), pruned_mask)
        with open(os.path.join(ckpt_dir, "bpas_state.json"), "w") as f:
            json.dump({str(k): v for k, v in prev_std_dict.items()}, f)

        total_pruned = int(pruned_mask.sum())
        active_batches = num_batches_total - total_pruned
        elapsed_hours = (time.time() - train_start_time) / 3600.0
        trainer_state = {
            "global_step": global_step,
            "epoch": round(display_epoch, 6),
            "epoch_integer": current_epoch + 1,  # 1-indexed for display
            "max_steps": max_steps,
            "learning_rate": scheduler.get_last_lr()[0],
            "total_sequences": num_train_sequences,
            "global_batch_size": global_batch_size,
            "total_batches": num_batches_total,
            "active_batches": active_batches,
            "active_sequences": active_batches * global_batch_size,
            "pruned_batches": total_pruned,
            "pruned_pct": round(100.0 * total_pruned / max(1, num_batches_total), 4),
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

        if save_total_limit and save_total_limit > 0:
            all_ckpts = sorted(
                [d for d in os.listdir(output_dir)
                 if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))],
                key=lambda x: int(x.split("-")[1]))
            while len(all_ckpts) > save_total_limit:
                old = all_ckpts.pop(0)
                shutil.rmtree(os.path.join(output_dir, old))
                logger.info(f"Deleted old checkpoint: {old}")

    if dist.is_initialized():
        dist.barrier()


def find_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    ckpts = [d for d in os.listdir(output_dir)
             if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
    if not ckpts:
        return None
    latest = max(ckpts, key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, latest)


def _load_state_dict(ckpt_dir, device):
    """Load the weights from pytorch_model.bin or model.safetensors."""
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    st_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location=device)
    if os.path.exists(st_path):
        from safetensors.torch import load_file
        return load_file(st_path)
    raise FileNotFoundError(f"No model weights in {ckpt_dir}")


def load_checkpoint(ckpt_dir, model, optimizer, scheduler, device):
    """Load model weights, optimizer, scheduler, and B-PAS state."""
    logger.info(f"Resuming from checkpoint: {ckpt_dir}")
    state_dict = _load_state_dict(ckpt_dir, device)
    model.module.load_state_dict(state_dict)

    ts = torch.load(os.path.join(ckpt_dir, "training_state.pt"), map_location=device)
    optimizer.load_state_dict(ts["optimizer"])
    scheduler.load_state_dict(ts["scheduler"])
    global_step = ts["global_step"]
    current_epoch = ts["current_epoch"]

    pruned_mask = np.load(os.path.join(ckpt_dir, "pruned_mask.npy"))

    prev_std_dict: Dict[int, float] = {}
    bpas_path = os.path.join(ckpt_dir, "bpas_state.json")
    if os.path.exists(bpas_path):
        with open(bpas_path) as f:
            raw = json.load(f)
            prev_std_dict = {int(k): float(v) for k, v in raw.items()}

    log_history = []
    ts_json_path = os.path.join(ckpt_dir, "trainer_state.json")
    if os.path.exists(ts_json_path):
        with open(ts_json_path) as f:
            log_history = json.load(f).get("log_history", [])

    logger.info(
        f"Resumed: step={global_step}, epoch={current_epoch + 1}, "
        f"pruned={int(pruned_mask.sum()):,} batches, "
        f"prev_std_dict={len(prev_std_dict):,} entries, "
        f"log_history={len(log_history)} entries")
    return global_step, current_epoch, pruned_mask, prev_std_dict, log_history


def run_epoch_end_bpas(epoch_std_buffer, batch_acc_buffer, prev_std_dict,
                       pruned_mask,
                       num_batches_total, current_epoch, total_epochs_est,
                       delta_start, delta_end, min_batches_fraction,
                       global_batch_size, local_batch_size,
                       tracking_dir, device, train_start_time):
    """End-of-epoch B-PAS step.

      1. Average each batch's activation std across ranks.
      2. Gather each batch's per-sequence next-token accuracy from all ranks (for the report only).
      3. From the second epoch on, prune batches whose std changed by at most the threshold.
      4. Rank 0 writes the per-batch report and the state needed to resume.
      5. Send the updated per-batch mask to all ranks.

    Returns (pruned_mask, current_std_dict).
    """
    is_dist = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_dist else 1
    rank = dist.get_rank() if is_dist else 0

    # Step 1: All-reduce stds across ranks (average)
    local_bids = sorted(epoch_std_buffer.keys())
    n_local = len(local_bids)

    if n_local > 0:
        stds_tensor = torch.tensor(
            [epoch_std_buffer[b] for b in local_bids],
            dtype=torch.float32, device=device)
        if is_dist:
            dist.all_reduce(stds_tensor, op=dist.ReduceOp.SUM)
            stds_tensor /= world_size
        current_std_dict: Dict[int, float] = {
            b: s for b, s in zip(local_bids, stds_tensor.tolist())}
    else:
        current_std_dict = {}

    # Step 1b: All-gather per-sequence accuracy across ranks
    # Each rank has batch_acc_buffer[bid] = np.array of shape (local_batch_size,).
    # All-gather to reconstruct the full (global_batch_size,) array per batch.
    global_batch_acc: Dict[int, np.ndarray] = {}
    if n_local > 0:
        local_acc_tensor = torch.zeros(
            n_local, local_batch_size, dtype=torch.float32, device=device)
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

    # Step 2: Compute threshold and prune
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

    # Step 3: Rank 0 writes stats
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
                f"GPU-hours={elapsed_h * world_size:.2f}\n\n")
            f.write(
                f"{'batch_id':>10}  {'mean_std':>12}  {'prev_std':>12}  "
                f"{'|delta|':>12}  {'action':>14}  "
                f"{'acc_mean':>10}  {'acc_median':>10}  "
                f"{'acc_min':>10}  {'acc_max':>10}  "
                f"{'pct_100':>10}\n")
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

                # Per-batch next-token accuracy stats from the global array.
                if bid in global_batch_acc:
                    acc_arr = global_batch_acc[bid]
                    valid = acc_arr[acc_arr >= 0]  # exclude -1 (no predicted positions)
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
                    f"{pct_100:>9.2f}%\n")

        logger.info(
            f"B-PAS epoch {current_epoch + 1}: threshold={threshold:.4e} | "
            f"newly_pruned={len(newly_pruned)} | total_pruned={total_pruned} | "
            f"active_batches={active_batches} "
            f"({active_batches * global_batch_size:,} seqs) | "
            f"GPU-hours={elapsed_h * world_size:.2f}")

        # Save the cumulative pruning mask + prev_std for resume.
        np.save(os.path.join(tracking_dir, "pruned_mask.npy"), pruned_mask)
        with open(os.path.join(tracking_dir, "bpas_prev_std_dict.json"), "w") as f:
            json.dump({str(k): v for k, v in current_std_dict.items()}, f)

    # Step 4: Broadcast pruned mask to all ranks
    if is_dist:
        pm_tensor = torch.from_numpy(pruned_mask.view(np.uint8)).to(device)
        dist.broadcast(pm_tensor, src=0)
        pruned_mask = pm_tensor.cpu().numpy().view(bool).copy()
        dist.barrier()

    return pruned_mask, current_std_dict


def train(cfg):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        logger.info(f"World size: {world_size}")
        logger.info(f"Config: {json.dumps(cfg, indent=2)}")

    set_seed(cfg["seed"])

    # Tokenizer (the vocabulary is fixed, so it is built here rather than loaded)
    sys.path.insert(0, MODEL_CODE_DIR)
    from tokenization_hyenadna import build_dna_tokenizer
    tokenizer = build_dna_tokenizer(model_max_length=cfg["max_seq_length"] + 2)

    # Data
    train_dataset, eval_dataset, num_train_sequences = prepare_datasets(
        cfg, tokenizer, rank, world_size)
    if rank == 0:
        logger.info(f"Training sequences: {num_train_sequences:,}")
        if eval_dataset is not None:
            logger.info(f"Eval sequences: {len(eval_dataset):,}")

    collator = CLMDataCollator(pad_token_id=tokenizer.pad_token_id)

    eval_dataloader = None
    if eval_dataset is not None and cfg.get("do_eval", True):
        eval_dataloader = make_eval_dataloader(eval_dataset, collator, cfg, rank, world_size)

    # Model
    model, model_config = build_model(cfg, tokenizer, device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Make the Hyena backbone record the output std of every block (see modeling_hyenadna.py)
    backbone = model.module.backbone
    backbone._collect_layer_stds = True
    num_layers = len(backbone.layers)
    if rank == 0:
        logger.info(
            f"B-PAS: activation std collection enabled on {num_layers} Hyena "
            f"blocks (std of the residual-stream representation per block)")

    # Optimizer: no weight decay on biases, norms, and all Hyena filter parameters
    no_decay_keys = ["bias", "norm", "filter_fn", "pos_emb", ".freq", "deltas"]
    decay_params, nodecay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nodecay_params if any(k in n for k in no_decay_keys) else decay_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": cfg["weight_decay"]},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=cfg["learning_rate"],
        betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
        eps=cfg["adam_epsilon"],
    )

    # Cosine learning-rate schedule with linear warmup over max_steps; not rescaled when the corpus shrinks
    max_steps = cfg["max_steps"]
    warmup_steps = cfg["warmup_steps"]
    sched_fn = (get_cosine_schedule_with_warmup
                if cfg["lr_scheduler_type"] == "cosine"
                else get_linear_schedule_with_warmup)
    scheduler = sched_fn(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)

    output_dir = cfg["output_dir"]
    tracking_dir = os.path.join(output_dir, "bpas_tracking")
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        os.makedirs(tracking_dir, exist_ok=True)

    global_batch_size = cfg["per_device_train_batch_size"] * world_size
    local_batch_size = cfg["per_device_train_batch_size"]

    # Deterministic shuffle + group (identical on all ranks).
    rng = np.random.default_rng(cfg["seed"])
    shuffled_idx = rng.permutation(num_train_sequences).astype(np.int64)

    num_complete_batches = num_train_sequences // global_batch_size
    num_used = num_complete_batches * global_batch_size
    num_tail = num_train_sequences - num_used

    if num_complete_batches == 0:
        raise ValueError(
            f"num_train_sequences={num_train_sequences} < global_batch_size="
            f"{global_batch_size}; cannot form a single fixed B-PAS batch.")

    # 2-D array: batch_groups[bid] = array of global_batch_size dataset indices.
    batch_groups = shuffled_idx[:num_used].reshape(num_complete_batches, global_batch_size)

    if rank == 0:
        logger.info(
            f"B-PAS: {num_train_sequences:,} sequences -> {num_complete_batches:,} "
            f"fixed global batches of {global_batch_size} "
            f"(per_device={local_batch_size} x world_size={world_size})")
        if num_tail > 0:
            logger.info(
                f"B-PAS: {num_tail:,} tail sequences dropped "
                f"({100.0 * num_tail / num_train_sequences:.4f}%)")
        # Which line_ids are in which batch; together with the per-batch mask this
        # gives the per-sequence picture
        np.save(os.path.join(tracking_dir, "batch_groups.npy"), batch_groups.astype(np.int32))

    batch_sampler = BPASBatchSampler(
        batch_groups=batch_groups, world_size=world_size, rank=rank,
        local_batch_size=local_batch_size, seed=cfg["seed"])

    # Number of epochs that fit in max_steps, used by the threshold schedule (fixed for the whole run)
    steps_per_epoch_initial = num_complete_batches // cfg["gradient_accumulation_steps"]
    total_epochs_est = (math.ceil(max_steps / steps_per_epoch_initial)
                        if steps_per_epoch_initial > 0 else 1)
    if rank == 0:
        logger.info(
            f"B-PAS: total_epochs_est={total_epochs_est} "
            f"(steps_per_epoch_initial={steps_per_epoch_initial}, max_steps={max_steps}). "
            f"It does not change as batches are pruned.")
        logger.info(
            f"B-PAS: delta_start={cfg['bpas_delta_start']:.4e}, "
            f"delta_end={cfg['bpas_delta_end']:.4e}, "
            f"min_batches_fraction={cfg['bpas_min_batches_fraction']}")

    # Pruning state
    pruned_mask = np.zeros(num_complete_batches, dtype=bool)
    prev_std_dict: Dict[int, float] = {}
    global_step = 0
    current_epoch = 0
    log_history = []

    # Resume from checkpoint
    ckpt_dir = find_latest_checkpoint(output_dir)
    if ckpt_dir is not None:
        global_step, current_epoch, pruned_mask, prev_std_dict, log_history = \
            load_checkpoint(ckpt_dir, model, optimizer, scheduler, device)
        if rank == 0:
            logger.info(
                f"Resuming: step {global_step}, epoch {current_epoch + 1}, "
                f"pruned {int(pruned_mask.sum()):,}/{num_complete_batches:,} batches")
    elif os.path.exists(os.path.join(tracking_dir, "pruned_mask.npy")):
        pruned_mask = np.load(os.path.join(tracking_dir, "pruned_mask.npy"))
        if rank == 0:
            logger.info(f"Loaded existing pruned_mask: {int(pruned_mask.sum()):,} batches pruned")
        bpas_std_path = os.path.join(tracking_dir, "bpas_prev_std_dict.json")
        if os.path.exists(bpas_std_path):
            with open(bpas_std_path) as f:
                raw = json.load(f)
                prev_std_dict = {int(k): float(v) for k, v in raw.items()}

    # Broadcast pruned_mask to all ranks.
    pm_tensor = torch.from_numpy(pruned_mask.view(np.uint8)).to(device)
    dist.broadcast(pm_tensor, src=0)
    pruned_mask = pm_tensor.cpu().numpy().view(bool).copy()

    active_ids = np.where(~pruned_mask)[0]
    batch_sampler.set_active_ids(active_ids)

    # Training config
    use_bf16 = cfg["bf16"]
    grad_accum = cfg["gradient_accumulation_steps"]
    max_grad_norm = cfg["max_grad_norm"]
    save_steps = cfg["save_steps"]
    eval_steps = cfg["eval_steps"]
    logging_steps = cfg["logging_steps"]
    save_total_limit = cfg["save_total_limit"]

    per_device_bs = cfg["per_device_train_batch_size"]
    effective_bs = per_device_bs * grad_accum * world_size
    if rank == 0:
        logger.info(f"Effective batch size: {per_device_bs} x {grad_accum} x {world_size} = {effective_bs}")
        logger.info(f"Max steps: {max_steps}, LR warmup: {warmup_steps}, scheduler: {cfg['lr_scheduler_type']}")
        logger.info(f"BF16: {use_bf16}")

    model.train()
    train_start_time = time.time()
    running_loss = 0.0
    log_step_count = 0
    optimizer.zero_grad()

    if rank == 0:
        logger.info("=" * 60)
        logger.info("TRAINING START")
        logger.info("=" * 60)

    while global_step < max_steps:
        # Update sampler active set (reflects pruning from previous epoch)
        active_ids = np.where(~pruned_mask)[0]
        batch_sampler.set_active_ids(active_ids)
        active_batch_count = len(active_ids)

        epoch_loader = make_train_dataloader(train_dataset, batch_sampler, collator, cfg)
        steps_this_epoch = len(epoch_loader) // grad_accum

        if rank == 0:
            logger.info(
                f"Epoch {current_epoch + 1}: {active_batch_count:,} active batches "
                f"({active_batch_count * global_batch_size:,} seqs), "
                f"{steps_this_epoch:,} steps/epoch "
                f"(pruned {int(pruned_mask.sum()):,}/{num_complete_batches:,})")

        # Early stopping: too few batches remaining.
        min_frac = cfg["bpas_min_batches_fraction"]
        if min_frac > 0 and active_batch_count < num_complete_batches * min_frac:
            if rank == 0:
                logger.info(
                    f"B-PAS early stop: only {active_batch_count} batches remain "
                    f"(< {min_frac*100:.1f}% of {num_complete_batches})")
            break

        # Stop if every batch has been pruned
        if active_batch_count == 0:
            if rank == 0:
                logger.info(
                    f"B-PAS: all {num_complete_batches} batches pruned. "
                    f"Saving final checkpoint at step {global_step}.")
            display_epoch = float(current_epoch)
            save_checkpoint(
                model, optimizer, scheduler, global_step, current_epoch,
                display_epoch, pruned_mask, prev_std_dict,
                num_complete_batches, num_train_sequences,
                global_batch_size, max_steps, log_history,
                output_dir, save_total_limit, tokenizer, rank,
                train_start_time, world_size, cfg)
            break

        # Per-epoch activation std buffer: {batch_id: mean_std}.
        epoch_std_buffer: Dict[int, float] = {}
        # Per-epoch accuracy buffer: {batch_id: np.array of per-seq next-token acc}.
        # Each rank stores its local_batch_size accuracies per batch; -1.0 = no
        # predicted positions.
        batch_acc_buffer: Dict[int, np.ndarray] = {}

        # batch_id order from the sampler (populated during __iter__).
        batch_ids_order: List[int] = []

        micro_step = 0
        for batch_enum_idx, batch in enumerate(epoch_loader):
            if global_step >= max_steps:
                break

            if batch_enum_idx == 0:
                batch_ids_order = batch_sampler.yielded_batch_ids
            bid = batch_ids_order[batch_enum_idx]

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])

            loss = outputs.loss
            if grad_accum > 1:
                loss = loss / grad_accum
            loss.backward()

            # Capture per-batch activation std (scalars from the backbone)
            layer_stds = backbone._layer_stds
            if layer_stds:
                mean_std = sum(layer_stds) / len(layer_stds)
                # Each bid appears exactly once per epoch (sampler guarantee).
                epoch_std_buffer[bid] = mean_std

            # Per-sequence next-token accuracy in this batch (for the report only)
            with torch.no_grad():
                logits = outputs.logits
                labels = batch["labels"]
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                preds = shift_logits.argmax(dim=-1)
                valid = shift_labels != -100
                per_seq_correct = ((preds == shift_labels) & valid).sum(dim=-1).float()
                per_seq_total = valid.sum(dim=-1).float()
                # accuracy per sequence; -1 if no predicted positions.
                per_seq_acc = torch.where(
                    per_seq_total > 0,
                    per_seq_correct / per_seq_total,
                    torch.tensor(-1.0, device=device))
                batch_acc_buffer[bid] = per_seq_acc.cpu().numpy()

            running_loss += outputs.loss.item()
            log_step_count += 1
            micro_step += 1

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                frac = micro_step / max(1, len(epoch_loader))
                display_epoch = current_epoch + frac

                if global_step % logging_steps == 0 and rank == 0:
                    avg_loss = running_loss / max(log_step_count, 1)
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - train_start_time
                    steps_per_sec = global_step / max(elapsed, 1)
                    eta_sec = (max_steps - global_step) / max(steps_per_sec, 1e-9)
                    logger.info(
                        f"step={global_step}/{max_steps} epoch={display_epoch:.2f} "
                        f"loss={avg_loss:.4f} lr={lr:.2e} speed={steps_per_sec:.1f} steps/s "
                        f"eta={eta_sec/3600:.1f}h")
                    log_history.append({
                        "step": global_step, "epoch": round(display_epoch, 4),
                        "loss": round(avg_loss, 6), "learning_rate": lr,
                        "active_batches": int(active_batch_count),
                        "pruned_batches": int(pruned_mask.sum()),
                        "steps_per_sec": round(steps_per_sec, 2),
                    })
                    running_loss = 0.0
                    log_step_count = 0

                if eval_dataloader is not None and global_step % eval_steps == 0:
                    metrics = evaluate(model, eval_dataloader, device, use_bf16)
                    if rank == 0:
                        logger.info(f"EVAL step={global_step}: loss={metrics['eval_loss']:.4f} "
                                    f"acc={metrics['eval_accuracy']:.4f} ppl={metrics['perplexity']:.2f}")
                        log_history.append({
                            "step": global_step, "epoch": round(display_epoch, 4),
                            "eval_loss": round(metrics["eval_loss"], 6),
                            "eval_accuracy": round(metrics["eval_accuracy"], 6),
                            "perplexity": round(metrics["perplexity"], 4),
                        })
                    model.train()

                if global_step % save_steps == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, global_step, current_epoch,
                        display_epoch, pruned_mask, prev_std_dict,
                        num_complete_batches, num_train_sequences,
                        global_batch_size, max_steps, log_history,
                        output_dir, save_total_limit, tokenizer, rank,
                        train_start_time, world_size, cfg)

        # End of epoch: compare activation stds and prune stable batches
        if global_step < max_steps:
            pruned_mask, prev_std_dict = run_epoch_end_bpas(
                epoch_std_buffer, batch_acc_buffer, prev_std_dict, pruned_mask,
                num_complete_batches, current_epoch, total_epochs_est,
                cfg["bpas_delta_start"], cfg["bpas_delta_end"],
                cfg["bpas_min_batches_fraction"],
                global_batch_size, local_batch_size,
                tracking_dir, device, train_start_time)
            if rank == 0:
                save_sequence_mask(os.path.join(tracking_dir, "pruned_sequence_mask.npy"),
                                   batch_groups, pruned_mask, num_train_sequences)
        current_epoch += 1

        # Stop after the scheduled number of epochs (counted from 1). The learning-rate
        # schedule is still tied to max_steps; the run just ends earlier. The folder
        # written here holds the final model only, without optimizer state.
        graceful_exit_epoch = cfg.get("graceful_exit_epoch", None)
        if graceful_exit_epoch and current_epoch == graceful_exit_epoch:
            if rank == 0:
                exit_dir = os.path.join(output_dir, f"epoch_{graceful_exit_epoch}_checkpoint")
                os.makedirs(exit_dir, exist_ok=True)
                model.module.save_pretrained(exit_dir, safe_serialization=False)
                tokenizer.save_pretrained(exit_dir)
                np.save(os.path.join(exit_dir, "pruned_mask.npy"), pruned_mask)
                with open(os.path.join(exit_dir, "bpas_state.json"), "w") as f:
                    json.dump({str(k): v for k, v in prev_std_dict.items()}, f)
                save_sequence_mask(os.path.join(exit_dir, "pruned_sequence_mask.npy"),
                                   batch_groups, pruned_mask, num_train_sequences)
                logger.info(
                    f"Epoch {graceful_exit_epoch} reached (step {global_step}), "
                    f"final model saved to {exit_dir}")
            dist.barrier()
            break

    total_time = (time.time() - train_start_time) / 3600.0
    total_pruned = int(pruned_mask.sum())
    if rank == 0:
        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE: {total_time:.2f}h wall-clock, {total_time*world_size:.2f} GPU-hours")
        logger.info(f"B-PAS final: {total_pruned}/{num_complete_batches} batches pruned "
                    f"({100.0*total_pruned/max(1,num_complete_batches):.2f}%)")
        logger.info("=" * 60)

        final_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        model.module.save_pretrained(final_dir, safe_serialization=False)
        tokenizer.save_pretrained(final_dir)
        np.save(os.path.join(final_dir, "pruned_mask.npy"), pruned_mask)
        save_sequence_mask(os.path.join(final_dir, "pruned_sequence_mask.npy"),
                           batch_groups, pruned_mask, num_train_sequences)
        logger.info(f"Final model saved to: {final_dir}")

    if eval_dataloader is not None:
        metrics = evaluate(model, eval_dataloader, device, use_bf16)
        if rank == 0:
            logger.info(f"FINAL EVAL: {metrics}")

    dist.destroy_process_group()


def main():
    if len(sys.argv) < 2:
        print("Usage: torchrun ... run_clm.py <train_config.json>")
        sys.exit(1)
    train(load_config(sys.argv[1]))


if __name__ == "__main__":
    main()
