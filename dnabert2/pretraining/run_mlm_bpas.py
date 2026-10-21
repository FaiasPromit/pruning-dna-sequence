#!/usr/bin/env python
"""
DNABERT-2 MLM pretraining with B-PAS (batch pruning by activation stability,
Alam et al., 2026), used as a dynamic-pruning baseline.

Before training, the corpus is split once into fixed global batches of
per_device_train_batch_size * world_size sequences, using the seed. A sequence
stays in the same batch for the whole run; only the order of the batches and
the order inside each batch are shuffled every epoch. The few sequences that do
not fill a complete global batch are not used.

For every batch, the standard deviation of each encoder layer's output is
recorded and averaged over the layers. From the second epoch on, a batch whose
average changed by at most delta(i) = delta_s * exp(alpha * i) since the
previous epoch is removed for the rest of the run, where
alpha = ln(delta_e / delta_s) / I and I is the number of full epochs that fit
in max_steps.

Sequences are numbered by line_id exactly as in run_mlm.py (their position
among the non-empty lines of train_file). The run saves batch_groups.npy, the
line_ids in each batch, and pruned_sequence_mask.npy, a per-sequence mask
indexed by line_id, so B-PAS runs can be compared sequence by sequence with
the other methods. pruned_mask.npy is the per-batch mask.

Everything else (model, optimizer, learning-rate schedule) is the same as in
run_mlm.py.

Launch with one process per GPU:
    torchrun --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> run_mlm_bpas.py <config.json>
"""

import json
import logging
import math
import os
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

import datasets as hf_datasets
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.optimization import get_linear_schedule_with_warmup

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# The DNABERT-2 model code (bert_layers.py and its helpers) lives in ../model_code
MODEL_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_code")
MODEL_CODE_FILES = ["bert_layers.py", "bert_padding.py", "flash_attn_triton.py", "configuration_bert.py"]
LOCAL_AUTO_MAP = {
    "AutoConfig": "configuration_bert.BertConfig",
    "AutoModel": "bert_layers.BertModel",
    "AutoModelForMaskedLM": "bert_layers.BertForMaskedLM",
    "AutoModelForSequenceClassification": "bert_layers.BertForSequenceClassification",
}


def add_model_code(model_dir):
    """Make a saved model folder self-contained (rank 0 only).

    Copies the model code from ../model_code into the folder and points auto_map in its
    config.json at those files. Without this, config.json points at the DNABERT-2 code on
    Hugging Face. With it, the folder can be fine-tuned with train.py or loaded with
    trust_remote_code=True as it is.
    """
    try:
        for name in MODEL_CODE_FILES:
            shutil.copy(os.path.join(MODEL_CODE_DIR, name), model_dir)
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        config["_name_or_path"] = "."
        config["auto_map"] = LOCAL_AUTO_MAP
        with open(config_path, "w") as f:
            f.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
    except Exception as e:
        logger.warning(f"Could not add the model code to {model_dir}: {e}")


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    # Defaults
    cfg.setdefault("gradient_accumulation_steps", 1)
    cfg.setdefault("max_grad_norm", 1.0)
    cfg.setdefault("dataloader_num_workers", 4)
    cfg.setdefault("dataloader_pin_memory", True)
    cfg.setdefault("dataloader_drop_last", True)
    cfg.setdefault("preprocessing_num_workers", 16)
    cfg.setdefault("save_steps", 10000)
    cfg.setdefault("eval_steps", 10000)
    cfg.setdefault("save_total_limit", 50)
    cfg.setdefault("logging_steps", 100)
    cfg.setdefault("seed", 42)
    cfg.setdefault("bf16", True)
    cfg.setdefault("mlm_probability", 0.15)
    cfg.setdefault("max_seq_length", 128)
    # B-PAS thresholds (delta_s, delta_e) and an optional early stop when few batches remain
    cfg.setdefault("bpas_delta_start", 1e-4)
    cfg.setdefault("bpas_delta_end", 1e-3)
    cfg.setdefault("bpas_min_batches_fraction", 0.0)
    cfg.setdefault("graceful_exit_epoch", None)  # same stopping rule as run_mlm.py
    return cfg


class MLMDataCollator(DataCollatorForLanguageModeling):
    """MLM collator that drops line_id if it is present, so it never reaches the model."""
    def __call__(self, features, return_tensors=None):
        for f in features:
            f.pop("line_id", None)
        return super().__call__(features, return_tensors=return_tensors)


def prepare_datasets(cfg, tokenizer, rank, world_size):
    """Load raw text and tokenize.

    Empty lines are dropped, so a sequence's index in train_dataset is its
    position among the non-empty lines of train_file: the same line_id that
    run_mlm.py uses.

    Returns (train_dataset, eval_dataset, num_train_sequences).
    train_dataset has columns: input_ids, attention_mask, special_tokens_mask
    """
    data_files = {}
    if cfg.get("train_file"):
        data_files["train"] = cfg["train_file"]
    if cfg.get("validation_file"):
        data_files["validation"] = cfg["validation_file"]

    max_seq_length = cfg["max_seq_length"]

    def tokenize_fn(examples):
        examples["text"] = [
            line for line in examples["text"]
            if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )

    num_proc = cfg["preprocessing_num_workers"]

    # Rank 0 preprocesses first; others wait and load from cache
    if rank == 0:
        logger.info("Rank 0: Loading and tokenizing dataset...")
        raw = load_dataset("text", data_files=data_files)
        tokenized = raw.map(
            tokenize_fn, batched=True, num_proc=num_proc,
            remove_columns=["text"], load_from_cache_file=True,
            desc="Tokenizing",
        )
        logger.info(f"Rank 0: dataset ready with {len(tokenized['train'])} train sequences")

    if world_size > 1:
        dist.barrier()

    if rank != 0:
        raw = load_dataset("text", data_files=data_files)
        tokenized = raw.map(
            tokenize_fn, batched=True, num_proc=num_proc,
            remove_columns=["text"], load_from_cache_file=True,
            desc="Tokenizing (from cache)",
        )

    train_dataset = tokenized["train"]
    eval_dataset = tokenized.get("validation", None)
    num_train_sequences = len(train_dataset)

    return train_dataset, eval_dataset, num_train_sequences


def build_model(cfg, tokenizer, device):
    """Create DNABERT-2 model from config, resize embeddings if needed."""
    config = AutoConfig.from_pretrained(
        cfg["config_name"],
        trust_remote_code=cfg.get("trust_remote_code", True),
    )

    sys.path.insert(0, MODEL_CODE_DIR)
    from bert_layers import BertForMaskedLM

    model = BertForMaskedLM(config)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != embedding_size:
        logger.info(f"Resizing embeddings: {embedding_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    model = model.to(device)
    return model, config


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
        """
        Args:
            batch_groups: array of shape (num_batches, global_batch_size);
                          batch_groups[bid] holds the line_ids in batch bid.
            world_size:   number of GPUs.
            rank:         this GPU's rank.
            local_batch_size: global_batch_size // world_size.
            seed:         RNG seed (same on all ranks).
        """
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

        # Shuffle batch order
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


def save_sequence_mask(path, batch_groups, pruned_mask, num_train_sequences):
    """Save a per-sequence mask indexed by line_id, like pruned_mask.npy in run_mlm.py.

    True marks a sequence that is no longer trained: it is in a pruned batch, or
    it is one of the leftover sequences that do not fill a complete global batch.
    """
    seq_mask = np.ones(num_train_sequences, dtype=bool)
    seq_mask[batch_groups[~pruned_mask].ravel()] = False
    np.save(path, seq_mask)


def make_train_dataloader(full_dataset, batch_sampler, data_collator, cfg):
    """Build a DataLoader driven by BPASBatchSampler.

    batch_sampler yields lists of dataset indices directly,
    so no batch_size, shuffle, sampler, or drop_last args are used.
    """
    return DataLoader(
        full_dataset,
        batch_sampler=batch_sampler,
        collate_fn=data_collator,
        num_workers=cfg["dataloader_num_workers"],
        pin_memory=cfg["dataloader_pin_memory"],
    )


def make_eval_dataloader(eval_dataset, tokenizer, cfg, rank, world_size):
    """Build eval DataLoader (no pruning)."""
    eval_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=cfg["mlm_probability"],
    )
    sampler = DistributedSampler(
        eval_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    return DataLoader(
        eval_dataset,
        batch_size=cfg.get("per_device_eval_batch_size", 128),
        sampler=sampler,
        collate_fn=eval_collator,
        num_workers=cfg["dataloader_num_workers"],
        pin_memory=cfg["dataloader_pin_memory"],
    )


@torch.no_grad()
def evaluate(model, eval_dataloader, device, use_bf16):
    """Run eval, return dict with loss, accuracy, perplexity."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    num_batches = 0

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(**batch)

        total_loss += outputs.loss.item()
        num_batches += 1

        # Accuracy on masked positions
        preds = outputs.logits.argmax(dim=-1)
        mask = labels != -100
        total_correct += ((preds == labels) & mask).sum().item()
        total_masked += mask.sum().item()

    # Aggregate across ranks
    stats = torch.tensor(
        [total_loss, num_batches, total_correct, total_masked],
        dtype=torch.float64, device=device,
    )
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

        # HF-compatible model checkpoint (works with from_pretrained)
        model.module.save_pretrained(ckpt_dir, safe_serialization=False)
        tokenizer.save_pretrained(ckpt_dir)
        add_model_code(ckpt_dir)

        # Training state (for optimizer/scheduler resume)
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "current_epoch": current_epoch,
        }, os.path.join(ckpt_dir, "training_state.pt"))

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
            "epoch_integer": current_epoch,
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

        # Also keep a copy at the output root for quick inspection
        with open(os.path.join(output_dir, "trainer_state.json"), "w") as f:
            json.dump(trainer_state, f, indent=2)

        logger.info(f"Saved checkpoint: {ckpt_dir}")

        # Rotate old checkpoints
        if save_total_limit and save_total_limit > 0:
            all_ckpts = sorted(
                [d for d in os.listdir(output_dir)
                 if d.startswith("checkpoint-") and
                 os.path.isdir(os.path.join(output_dir, d))],
                key=lambda x: int(x.split("-")[1]),
            )
            while len(all_ckpts) > save_total_limit:
                old = all_ckpts.pop(0)
                shutil.rmtree(os.path.join(output_dir, old))
                logger.info(f"Deleted old checkpoint: {old}")

    # All ranks wait for rank 0 to finish saving
    if dist.is_initialized():
        dist.barrier()


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint-NNNNN directory."""
    if not os.path.isdir(output_dir):
        return None
    ckpts = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and
        os.path.isdir(os.path.join(output_dir, d))
    ]
    if not ckpts:
        return None
    latest = max(ckpts, key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, latest)


def load_checkpoint(ckpt_dir, model, optimizer, scheduler, device):
    """Load model weights, optimizer, scheduler, and B-PAS state."""
    logger.info(f"Resuming from checkpoint: {ckpt_dir}")

    # Model weights
    weights_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location=device)
    model.module.load_state_dict(state_dict)

    # Optimizer + scheduler + step counter
    ts = torch.load(os.path.join(ckpt_dir, "training_state.pt"), map_location=device)
    optimizer.load_state_dict(ts["optimizer"])
    scheduler.load_state_dict(ts["scheduler"])
    global_step = ts["global_step"]
    current_epoch = ts["current_epoch"]

    # Pruning mask (per-batch boolean)
    pruned_mask = np.load(os.path.join(ckpt_dir, "pruned_mask.npy"))

    # Previous epoch std dict
    prev_std_dict: Dict[int, float] = {}
    bpas_path = os.path.join(ckpt_dir, "bpas_state.json")
    if os.path.exists(bpas_path):
        with open(bpas_path) as f:
            raw = json.load(f)
            prev_std_dict = {int(k): float(v) for k, v in raw.items()}

    # Log history
    log_history = []
    ts_json_path = os.path.join(ckpt_dir, "trainer_state.json")
    if os.path.exists(ts_json_path):
        with open(ts_json_path) as f:
            ts_json = json.load(f)
            log_history = ts_json.get("log_history", [])

    logger.info(
        f"Resumed: step={global_step}, epoch={current_epoch}, "
        f"pruned={int(pruned_mask.sum()):,} batches, "
        f"prev_std_dict={len(prev_std_dict):,} entries, "
        f"log_history={len(log_history)} entries"
    )
    return global_step, current_epoch, pruned_mask, prev_std_dict, log_history


def run_epoch_end_bpas(epoch_std_buffer, batch_acc_buffer, prev_std_dict,
                       pruned_mask,
                       num_batches_total, current_epoch, total_epochs_est,
                       delta_start, delta_end, min_batches_fraction,
                       global_batch_size, local_batch_size,
                       tracking_dir, device,
                       train_start_time):
    """End-of-epoch B-PAS step.

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
    global_batch_acc: Dict[int, np.ndarray] = {}  # batch id -> (global_batch_size,)
    if n_local > 0:
        # Pack local accuracies into an (n_local, local_batch_size) tensor
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
            # Concatenate the ranks' slices into (n_local, global_batch_size)
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
    threshold = bpas_threshold(delta_start, delta_end, total_epochs_est,
                               current_epoch)

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
            tracking_dir, f"epoch_{current_epoch:04d}_bpas_stats.txt")
        with open(stats_path, "w") as f:
            f.write(
                f"B-PAS Epoch {current_epoch} | "
                f"threshold={threshold:.4e} | "
                f"newly_pruned={len(newly_pruned)} batches | "
                f"total_pruned={total_pruned}/{num_batches_total} | "
                f"active={active_batches} batches "
                f"({active_batches * global_batch_size:,} seqs) | "
                f"GPU-hours={elapsed_h * world_size:.2f}\n\n"
            )
            # Header, including per-batch accuracy columns
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
            f"B-PAS epoch {current_epoch}: threshold={threshold:.4e} | "
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


def train(cfg):
    # Distributed setup
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=cfg.get("ddp_timeout", 7200)),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        logger.info(f"World size: {world_size}")
        logger.info(f"Config: {json.dumps(cfg, indent=2)}")

    set_seed(cfg["seed"])

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["tokenizer_name"],
        trust_remote_code=cfg.get("trust_remote_code", True),
    )

    # Data
    train_dataset, eval_dataset, num_train_sequences = prepare_datasets(
        cfg, tokenizer, rank, world_size,
    )
    if rank == 0:
        logger.info(f"Training sequences: {num_train_sequences:,}")
        if eval_dataset is not None:
            logger.info(f"Eval sequences: {len(eval_dataset):,}")

    train_collator = MLMDataCollator(
        tokenizer=tokenizer,
        mlm_probability=cfg["mlm_probability"],
    )

    eval_dataloader = None
    if eval_dataset is not None and cfg.get("do_eval", True):
        eval_dataloader = make_eval_dataloader(
            eval_dataset, tokenizer, cfg, rank, world_size,
        )

    # Model
    model, model_config = build_model(cfg, tokenizer, device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Turn on per-layer activation std collection in the encoder (see bert_layers.py)
    encoder = model.module.bert.encoder
    encoder._collect_layer_stds = True
    num_layers = len(encoder.layer)
    if rank == 0:
        logger.info(
            f"B-PAS: activation std collection enabled on {num_layers} "
            f"encoder layers (unpadded 2-D hidden states)"
        )

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight", "layer_norm.bias"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.get("weight_decay", 1e-5),
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg["learning_rate"],
        betas=(cfg.get("adam_beta1", 0.9), cfg.get("adam_beta2", 0.98)),
        eps=cfg.get("adam_epsilon", 1e-6),
    )

    # Linear warmup and decay over max_steps; not rescaled when the corpus shrinks
    max_steps = cfg["max_steps"]
    warmup_steps = cfg.get("warmup_steps", 0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    # Output folders
    output_dir = cfg["output_dir"]
    tracking_dir = os.path.join(output_dir, "bpas_tracking")
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        os.makedirs(tracking_dir, exist_ok=True)

    global_batch_size = cfg["per_device_train_batch_size"] * world_size
    local_batch_size = cfg["per_device_train_batch_size"]

    # Fixed global batches, assigned once from the seed (the same on all ranks)
    rng = np.random.default_rng(cfg["seed"])
    shuffled_idx = rng.permutation(num_train_sequences).astype(np.int64)

    num_complete_batches = num_train_sequences // global_batch_size
    num_used = num_complete_batches * global_batch_size
    num_tail = num_train_sequences - num_used

    # batch_groups[bid] holds the global_batch_size line_ids of batch bid
    batch_groups = shuffled_idx[:num_used].reshape(num_complete_batches,
                                                    global_batch_size)

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

    # B-PAS batch sampler
    batch_sampler = BPASBatchSampler(
        batch_groups=batch_groups,
        world_size=world_size,
        rank=rank,
        local_batch_size=local_batch_size,
        seed=cfg["seed"],
    )

    # Number of epochs I in the threshold schedule, estimated from max_steps
    steps_per_epoch_initial = num_complete_batches // cfg["gradient_accumulation_steps"]
    total_epochs_est = (
        math.ceil(max_steps / steps_per_epoch_initial)
        if steps_per_epoch_initial > 0 else 1
    )
    if rank == 0:
        logger.info(
            f"B-PAS: total_epochs_est={total_epochs_est} "
            f"(steps_per_epoch_initial={steps_per_epoch_initial}, "
            f"max_steps={max_steps}). "
            f"Fixed for the whole run."
        )
        logger.info(
            f"B-PAS: delta_start={cfg['bpas_delta_start']:.4e}, "
            f"delta_end={cfg['bpas_delta_end']:.4e}, "
            f"min_batches_fraction={cfg['bpas_min_batches_fraction']}"
        )

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
                f"Resuming: step {global_step}, epoch {current_epoch}, "
                f"pruned {int(pruned_mask.sum()):,}/{num_complete_batches:,} batches"
            )
    elif os.path.exists(os.path.join(tracking_dir, "pruned_mask.npy")):
        # No full checkpoint but pruning mask exists (edge case)
        pruned_mask = np.load(os.path.join(tracking_dir, "pruned_mask.npy"))
        if rank == 0:
            logger.info(f"Loaded existing pruned_mask: {int(pruned_mask.sum()):,} batches pruned")
        bpas_std_path = os.path.join(tracking_dir, "bpas_prev_std_dict.json")
        if os.path.exists(bpas_std_path):
            with open(bpas_std_path) as f:
                raw = json.load(f)
                prev_std_dict = {int(k): float(v) for k, v in raw.items()}

    # Broadcast pruned_mask to all ranks
    pm_tensor = torch.from_numpy(pruned_mask.view(np.uint8)).to(device)
    dist.broadcast(pm_tensor, src=0)
    pruned_mask = pm_tensor.cpu().numpy().view(bool).copy()

    # Update sampler with current active set
    active_ids = np.where(~pruned_mask)[0]
    batch_sampler.set_active_ids(active_ids)

    # Training config
    use_bf16 = cfg.get("bf16", True)
    grad_accum = cfg["gradient_accumulation_steps"]
    max_grad_norm = cfg["max_grad_norm"]
    save_steps = cfg["save_steps"]
    eval_steps = cfg["eval_steps"]
    logging_steps = cfg.get("logging_steps", 100)
    save_total_limit = cfg.get("save_total_limit", 50)

    # Effective batch size logging
    per_device_bs = cfg["per_device_train_batch_size"]
    effective_bs = per_device_bs * grad_accum * world_size
    if rank == 0:
        logger.info(f"Effective batch size: {per_device_bs} x {grad_accum} x {world_size} = {effective_bs}")
        logger.info(f"Max steps: {max_steps}, warmup: {warmup_steps}")
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
        # Active batches for this epoch
        active_ids = np.where(~pruned_mask)[0]
        batch_sampler.set_active_ids(active_ids)
        active_batch_count = len(active_ids)

        # Build a fresh dataloader for this epoch
        epoch_loader = make_train_dataloader(
            train_dataset, batch_sampler, train_collator, cfg,
        )
        steps_this_epoch = len(epoch_loader) // grad_accum

        if rank == 0:
            logger.info(
                f"Epoch {current_epoch}: {active_batch_count:,} active batches "
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
                model, optimizer, scheduler, global_step, current_epoch,
                display_epoch, pruned_mask, prev_std_dict,
                num_complete_batches, num_train_sequences,
                global_batch_size, max_steps, log_history,
                output_dir, save_total_limit, tokenizer, rank,
                train_start_time, world_size, cfg,
            )
            break

        # Per-epoch activation std of each batch: {batch_id: mean_std}
        epoch_std_buffer: Dict[int, float] = {}

        # Per-epoch MLM accuracy of this rank's sequences in each batch
        # (-1.0 means the sequence had no [MASK] positions)
        batch_acc_buffer: Dict[int, np.ndarray] = {}

        batch_ids_order: List[int] = []

        micro_step = 0
        for batch_enum_idx, batch in enumerate(epoch_loader):
            if global_step >= max_steps:
                break

            # The sampler records this epoch's batch order once iteration starts
            if batch_enum_idx == 0:
                batch_ids_order = batch_sampler.yielded_batch_ids

            # Current batch_id
            bid = batch_ids_order[batch_enum_idx]

            # Move to GPU
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Forward pass
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(**batch)

            loss = outputs.loss
            if grad_accum > 1:
                loss = loss / grad_accum
            loss.backward()

            # Mean of the per-layer output stds for this batch
            layer_stds = encoder._layer_stds
            if layer_stds:
                mean_std = sum(layer_stds) / len(layer_stds)
                # Each batch appears exactly once per epoch
                epoch_std_buffer[bid] = mean_std

            # Per-sequence MLM accuracy within this batch (for the report)
            with torch.no_grad():
                preds = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
                mask_pos = labels != -100
                per_seq_correct = ((preds == labels) & mask_pos).sum(dim=-1).float()
                per_seq_total = mask_pos.sum(dim=-1).float()
                # -1 if the sequence has no [MASK] positions
                per_seq_acc = torch.where(
                    per_seq_total > 0,
                    per_seq_correct / per_seq_total,
                    torch.tensor(-1.0, device=device),
                )
                batch_acc_buffer[bid] = per_seq_acc.cpu().numpy()

            running_loss += outputs.loss.item()
            log_step_count += 1
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
                if global_step % logging_steps == 0 and rank == 0:
                    avg_loss = running_loss / max(log_step_count, 1)
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - train_start_time
                    steps_per_sec = global_step / max(elapsed, 1)
                    eta_sec = (max_steps - global_step) / max(steps_per_sec, 1e-9)
                    logger.info(
                        f"step={global_step}/{max_steps} "
                        f"epoch={display_epoch:.2f} "
                        f"loss={avg_loss:.4f} "
                        f"lr={lr:.2e} "
                        f"speed={steps_per_sec:.1f} steps/s "
                        f"eta={eta_sec/3600:.1f}h"
                    )
                    log_history.append({
                        "step": global_step,
                        "epoch": round(display_epoch, 4),
                        "loss": round(avg_loss, 6),
                        "learning_rate": lr,
                        "active_batches": int(active_batch_count),
                        "pruned_batches": int(pruned_mask.sum()),
                        "steps_per_sec": round(steps_per_sec, 2),
                    })
                    running_loss = 0.0
                    log_step_count = 0

                # Eval
                if eval_dataloader is not None and global_step % eval_steps == 0:
                    metrics = evaluate(model, eval_dataloader, device, use_bf16)
                    if rank == 0:
                        logger.info(
                            f"EVAL step={global_step}: "
                            f"loss={metrics['eval_loss']:.4f} "
                            f"acc={metrics['eval_accuracy']:.4f} "
                            f"ppl={metrics['perplexity']:.2f}"
                        )
                        log_history.append({
                            "step": global_step,
                            "epoch": round(display_epoch, 4),
                            "eval_loss": round(metrics["eval_loss"], 6),
                            "eval_accuracy": round(metrics["eval_accuracy"], 6),
                            "perplexity": round(metrics["perplexity"], 4),
                        })
                    model.train()

                # Save
                if global_step % save_steps == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, global_step, current_epoch,
                        display_epoch, pruned_mask, prev_std_dict,
                        num_complete_batches, num_train_sequences,
                        global_batch_size, max_steps, log_history,
                        output_dir, save_total_limit, tokenizer, rank,
                        train_start_time, world_size, cfg,
                    )

        # End of epoch: compare activation stds and prune stable batches
        if global_step < max_steps:
            pruned_mask, prev_std_dict = run_epoch_end_bpas(
                epoch_std_buffer, batch_acc_buffer, prev_std_dict, pruned_mask,
                num_complete_batches, current_epoch, total_epochs_est,
                cfg["bpas_delta_start"], cfg["bpas_delta_end"],
                cfg["bpas_min_batches_fraction"],
                global_batch_size, local_batch_size,
                tracking_dir, device,
                train_start_time,
            )
            if rank == 0:
                save_sequence_mask(os.path.join(tracking_dir, "pruned_sequence_mask.npy"),
                                   batch_groups, pruned_mask, num_train_sequences)
        current_epoch += 1

        # Stop after the scheduled number of epochs
        graceful_exit_epoch = cfg.get("graceful_exit_epoch", None)
        if graceful_exit_epoch and current_epoch == graceful_exit_epoch:
            if rank == 0:
                exit_dir = os.path.join(output_dir, f"epoch_{graceful_exit_epoch}_checkpoint")
                os.makedirs(exit_dir, exist_ok=True)
                model.module.save_pretrained(exit_dir, safe_serialization=False)
                tokenizer.save_pretrained(exit_dir)
                add_model_code(exit_dir)
                np.save(os.path.join(exit_dir, "pruned_mask.npy"), pruned_mask)
                save_sequence_mask(os.path.join(exit_dir, "pruned_sequence_mask.npy"),
                                   batch_groups, pruned_mask, num_train_sequences)
                logger.info(f"Epoch {graceful_exit_epoch} reached, final model saved to {exit_dir}")
            dist.barrier()
            break

    total_time = (time.time() - train_start_time) / 3600.0
    total_pruned = int(pruned_mask.sum())
    if rank == 0:
        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE: {total_time:.2f}h wall-clock, "
                     f"{total_time * world_size:.2f} GPU-hours")
        logger.info(f"B-PAS final: {total_pruned}/{num_complete_batches} batches pruned "
                     f"({100.0*total_pruned/max(1,num_complete_batches):.2f}%)")
        logger.info("=" * 60)

        # Final save
        final_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        model.module.save_pretrained(final_dir, safe_serialization=False)
        tokenizer.save_pretrained(final_dir)
        add_model_code(final_dir)
        np.save(os.path.join(final_dir, "pruned_mask.npy"), pruned_mask)
        save_sequence_mask(os.path.join(final_dir, "pruned_sequence_mask.npy"),
                           batch_groups, pruned_mask, num_train_sequences)
        logger.info(f"Final model saved to: {final_dir}")

    # Final eval
    if eval_dataloader is not None:
        metrics = evaluate(model, eval_dataloader, device, use_bf16)
        if rank == 0:
            logger.info(f"FINAL EVAL: {metrics}")

    dist.destroy_process_group()


def main():
    if len(sys.argv) < 2:
        print("Usage: torchrun ... run_mlm_bpas.py <config.json>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    train(cfg)


if __name__ == "__main__":
    main()
