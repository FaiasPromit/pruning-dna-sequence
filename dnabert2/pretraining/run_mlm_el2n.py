#!/usr/bin/env python
"""
DNABERT-2 MLM pretraining with sequence pruning by EL2N instead of accuracy.

EL2N (Paul et al., 2021) is computed for each sequence from the same training
forward pass: at each [MASK] position, the L2 norm of (softmax prediction minus
the one-hot target), averaged over the sequence's [MASK] positions. Low EL2N
means the sequence is already predicted well, so the pruning range starts at 0:
at each epoch boundary, active sequences with EL2N in
[pruning_lower_threshold, pruning_upper_threshold] are removed for the rest of
the run. Everything else is the same as in run_mlm.py.

Launch with one process per GPU:
    torchrun --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> run_mlm_el2n.py <config.json>
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

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
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

# EL2N = ||softmax(logits) - onehot(label)||_2, averaged over a sequence's [MASK]
# positions. It lies in [0, sqrt(2)]: 0 is a perfect prediction, about 1.0 a uniform
# prediction over the vocabulary, and sqrt(2) a confident wrong prediction.
# Low EL2N therefore marks an easy sequence (the opposite direction to accuracy).
EL2N_MAX = 2.0 ** 0.5
SCORE_NOT_TRAINED = -1.0   # active but not processed this epoch
SCORE_NO_MASK = -2.0       # processed but had no [MASK] positions
EL2N_BIN_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                  0.9, 1.0, 1.1, 1.2, 1.3, EL2N_MAX + 1e-6]


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
    cfg.setdefault("pruning_enabled", True)
    # Pruning range for EL2N. Because low EL2N means easy, the range starts at 0
    # and its upper end is the tunable threshold (accuracy fixes the upper end at
    # 1 and tunes the lower end instead).
    cfg.setdefault("pruning_lower_threshold", 0.00)
    cfg.setdefault("pruning_upper_threshold", 0.70)
    cfg.setdefault("graceful_exit_epoch", None)
    return cfg


class MLMTrackingDataCollator(DataCollatorForLanguageModeling):
    """MLM collator that keeps each sequence's line_id (its index in the corpus).

    line_id is removed before masking and added back to the batch afterwards.
    """
    def __call__(self, features, return_tensors=None):
        line_ids = [f.pop("line_id", -1) for f in features]
        batch = super().__call__(features, return_tensors=return_tensors)
        batch["line_id"] = torch.tensor(line_ids, dtype=torch.long)
        return batch


def prepare_datasets(cfg, tokenizer, rank, world_size):
    """Load raw text, tokenize, and give each training sequence a line_id.

    A sequence's line_id is its position among the non-empty lines of
    train_file, counting from 0. It depends only on the training file, never on
    the pruning settings, so the same line_id refers to the same sequence in
    every run on that corpus, and all saved per-sequence arrays are indexed by it.

    Returns (train_dataset, eval_dataset, num_train_sequences).
    train_dataset has columns: input_ids, attention_mask, special_tokens_mask, line_id
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

    def add_line_ids(example, idx):
        return {"line_id": idx}

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
        if "train" in tokenized:
            tokenized["train"] = tokenized["train"].map(
                add_line_ids, with_indices=True, num_proc=num_proc,
                load_from_cache_file=True, desc="Adding line IDs",
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
        if "train" in tokenized:
            tokenized["train"] = tokenized["train"].map(
                add_line_ids, with_indices=True, num_proc=num_proc,
                load_from_cache_file=True, desc="Adding line IDs (from cache)",
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


def make_train_dataloader(full_dataset, pruned_mask, data_collator, cfg,
                          rank, world_size, epoch):
    """Build a fresh DataLoader over only the active (unpruned) sequences."""
    active_indices = np.where(~pruned_mask)[0].tolist()

    subset = Subset(full_dataset, active_indices)
    sampler = DistributedSampler(
        subset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
        seed=cfg["seed"],
    )
    sampler.set_epoch(epoch)

    loader = DataLoader(
        subset,
        batch_size=cfg["per_device_train_batch_size"],
        sampler=sampler,
        collate_fn=data_collator,
        num_workers=cfg["dataloader_num_workers"],
        pin_memory=cfg["dataloader_pin_memory"],
        drop_last=cfg["dataloader_drop_last"],
    )
    return loader, len(active_indices)


def make_eval_dataloader(eval_dataset, tokenizer, cfg, rank, world_size):
    """Build eval DataLoader (no pruning, no line_id tracking)."""
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
                    display_epoch, pruned_mask, epoch_buffer, no_mask_buffer,
                    num_train_sequences, max_steps, log_history,
                    output_dir, save_total_limit, tokenizer, rank,
                    train_start_time, world_size):
    """Save the model, optimizer and scheduler state, pruning state, and a run summary (rank 0 only)."""
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

        # Pruning state
        np.save(os.path.join(ckpt_dir, "pruned_mask.npy"), pruned_mask)
        with open(os.path.join(ckpt_dir, "pruning_buffer.json"), "w") as f:
            json.dump({
                "epoch_buffer": {str(k): [float(v[0]), int(v[1])] for k, v in epoch_buffer.items()},
                "no_mask_buffer": list(no_mask_buffer),
            }, f)

        # trainer_state.json: human-readable run summary
        total_pruned = int(pruned_mask.sum())
        elapsed_hours = (time.time() - train_start_time) / 3600.0
        trainer_state = {
            "global_step": global_step,
            "epoch": round(display_epoch, 6),
            "epoch_integer": current_epoch,
            "max_steps": max_steps,
            "learning_rate": scheduler.get_last_lr()[0],
            "total_sequences": num_train_sequences,
            "active_sequences": num_train_sequences - total_pruned,
            "total_pruned": total_pruned,
            "pruned_pct": round(100.0 * total_pruned / max(1, num_train_sequences), 4),
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
    """Load model weights, optimizer, scheduler, and pruning state."""
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

    # Pruning mask
    pruned_mask = np.load(os.path.join(ckpt_dir, "pruned_mask.npy"))

    # Log history
    log_history = []
    ts_json_path = os.path.join(ckpt_dir, "trainer_state.json")
    if os.path.exists(ts_json_path):
        with open(ts_json_path) as f:
            ts_json = json.load(f)
            log_history = ts_json.get("log_history", [])

    logger.info(
        f"Resumed: step={global_step}, epoch={current_epoch}, "
        f"pruned={int(pruned_mask.sum()):,}, log_history={len(log_history)} entries"
    )
    return global_step, current_epoch, pruned_mask, log_history


def run_epoch_end_pruning(epoch_buffer, no_mask_buffer, pruned_mask,
                          num_train_sequences, current_epoch,
                          tracking_dir, device, train_start_time, cfg):
    """Gather per-sequence EL2N from all ranks and update the pruning mask.

    Sequences whose EL2N lies in [pruning_lower_threshold, pruning_upper_threshold]
    are marked as pruned. In el2n_array, -1 means the sequence was not processed
    this epoch and -2 means it had no [MASK] positions; neither is ever pruned.
    Returns (updated_pruned_mask, el2n_array), where el2n_array is None on ranks other than 0.
    """
    is_distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_distributed else 1
    rank = dist.get_rank() if is_distributed else 0

    # Pack local tracking data into tensors
    local_lids = list(epoch_buffer.keys())
    local_el2n = [epoch_buffer[k][0] for k in local_lids]
    local_total = [epoch_buffer[k][1] for k in local_lids]

    t_lids = torch.tensor(local_lids or [0], dtype=torch.long, device=device)
    t_el2n = torch.tensor(local_el2n or [0.0], dtype=torch.float64, device=device)
    t_total = torch.tensor(local_total or [0], dtype=torch.float64, device=device)
    nm_lids = torch.tensor(list(no_mask_buffer) or [0], dtype=torch.long, device=device)

    actual_lid_count = len(local_lids)
    actual_nm_count = len(no_mask_buffer)
    local_size = torch.tensor([actual_lid_count], dtype=torch.long, device=device)
    nm_local_size = torch.tensor([actual_nm_count], dtype=torch.long, device=device)

    if is_distributed:
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        nm_all_sizes = [torch.zeros_like(nm_local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        dist.all_gather(nm_all_sizes, nm_local_size)

        max_size = max(s.item() for s in all_sizes) if all_sizes else 1
        nm_max_size = max(s.item() for s in nm_all_sizes) if nm_all_sizes else 1
        max_size = max(max_size, 1)
        nm_max_size = max(nm_max_size, 1)

        t_lids_padded = torch.zeros(max_size, dtype=torch.long, device=device)
        t_el2n_padded = torch.zeros(max_size, dtype=torch.float64, device=device)
        t_total_padded = torch.zeros(max_size, dtype=torch.float64, device=device)
        nm_lids_padded = torch.zeros(nm_max_size, dtype=torch.long, device=device)

        t_lids_padded[:actual_lid_count] = t_lids[:actual_lid_count]
        t_el2n_padded[:actual_lid_count] = t_el2n[:actual_lid_count]
        t_total_padded[:actual_lid_count] = t_total[:actual_lid_count]
        nm_lids_padded[:actual_nm_count] = nm_lids[:actual_nm_count]

        all_lids = [torch.zeros_like(t_lids_padded) for _ in range(world_size)]
        all_el2n = [torch.zeros_like(t_el2n_padded) for _ in range(world_size)]
        all_total = [torch.zeros_like(t_total_padded) for _ in range(world_size)]
        nm_all_lids = [torch.zeros_like(nm_lids_padded) for _ in range(world_size)]

        dist.all_gather(all_lids, t_lids_padded)
        dist.all_gather(all_el2n, t_el2n_padded)
        dist.all_gather(all_total, t_total_padded)
        dist.all_gather(nm_all_lids, nm_lids_padded)
    else:
        all_sizes = [local_size]
        nm_all_sizes = [nm_local_size]
        all_lids = [t_lids]
        all_el2n = [t_el2n]
        all_total = [t_total]
        nm_all_lids = [nm_lids]

    # Merge on rank 0 and update pruning mask
    if rank == 0:
        display_epoch = current_epoch + 1  # 1-indexed for consistency with trainer_state
        merged = {}
        for r in range(world_size):
            sz = all_sizes[r].item()
            r_lids = all_lids[r][:sz].cpu().numpy()
            r_el2n = all_el2n[r][:sz].cpu().numpy()
            r_total = all_total[r][:sz].cpu().numpy()
            for lid, c, t in zip(r_lids, r_el2n, r_total):
                lid = int(lid)
                if lid in merged:
                    merged[lid] = (merged[lid][0] + c, merged[lid][1] + t)
                else:
                    merged[lid] = (c, t)

        el2n_array = np.full(num_train_sequences, SCORE_NOT_TRAINED, dtype=np.float64)
        for lid, (c, t) in merged.items():
            el2n_array[lid] = SCORE_NO_MASK if t <= 0 else float(c) / float(t)

        no_mask_set = set()
        for r in range(world_size):
            sz = nm_all_sizes[r].item()
            for lid in nm_all_lids[r][:sz].cpu().numpy():
                no_mask_set.add(int(lid))
        for lid in no_mask_set:
            if el2n_array[lid] == SCORE_NOT_TRAINED:
                el2n_array[lid] = SCORE_NO_MASK

        # Stats on currently-active sequences
        keep_before = ~pruned_mask
        current_line_ids = np.where(keep_before)[0]
        current_active_count = len(current_line_ids)
        current_active_el2n = el2n_array[current_line_ids]
        # counted the same way as in run_mlm.py
        tracked_count = len(merged) + int((el2n_array == SCORE_NO_MASK).sum())

        os.makedirs(tracking_dir, exist_ok=True)

        dnt_count = int((current_active_el2n == SCORE_NOT_TRAINED).sum())
        nm_count = int((current_active_el2n == SCORE_NO_MASK).sum())
        scored_vals = current_active_el2n[current_active_el2n >= 0.0]

        # Write EL2N distribution
        stats_path = os.path.join(tracking_dir, f"epoch_{display_epoch}_el2n_dist.txt")
        with open(stats_path, "w") as f:
            f.write(f"Epoch {display_epoch} Per-Sequence EL2N Distribution\n")
            f.write(f"Current active dataset size: {current_active_count}\n")
            f.write(f"Processed/tracked this epoch: {tracked_count}\n")
            f.write(f"Scored active sequences: {scored_vals.size}\n\n")
            f.write(f"{'Bucket':>20} {'% of active':>14} {'count':>14}\n")
            f.write("-" * 52 + "\n")
            f.write(f"{'DID NOT TRAIN':>20} {100.0*dnt_count/max(1,current_active_count):13.2f}% {dnt_count:14,}\n")
            f.write(f"{'NO MASKED TOKENS':>20} {100.0*nm_count/max(1,current_active_count):13.2f}% {nm_count:14,}\n")

            if scored_vals.size:
                for i in range(len(EL2N_BIN_EDGES) - 1):
                    lo, hi = EL2N_BIN_EDGES[i], EL2N_BIN_EDGES[i + 1]
                    count = int(((scored_vals >= lo) & (scored_vals < hi)).sum())
                    f.write(f"{f'[{lo:.1f}, {hi:.1f})':>20} "
                            f"{100.0*count/max(1,current_active_count):13.2f}% {count:14,}\n")

                f.write("\nPercentiles over scored active sequences:\n")
                f.write("-" * 52 + "\n")
                for pc in [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]:
                    f.write(f"{'p'+str(pc):>20} {float(np.percentile(scored_vals, pc)):>20.6f}\n")
                f.write(f"{'mean':>20} {float(scored_vals.mean()):>20.6f}\n")
                f.write(f"{'std':>20} {float(scored_vals.std()):>20.6f}\n")

                # How many active sequences each candidate upper threshold would prune now
                f.write("\nCalibration -- what each upper threshold would prune NOW:\n")
                f.write("-" * 52 + "\n")
                for t in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]:
                    c = int((scored_vals <= t).sum())
                    f.write(f"{'<= ' + f'{t:.2f}':>20} "
                            f"{100.0*c/max(1,current_active_count):13.2f}% {c:14,}\n")

        # Prune sequences within configured threshold range
        pruning_enabled = cfg.get("pruning_enabled", True)
        prune_lo = cfg.get("pruning_lower_threshold", 0.00)
        prune_hi = cfg.get("pruning_upper_threshold", 0.70)

        if pruning_enabled:
            # The -1 and -2 markers are below any lower threshold >= 0, so those
            # sequences are never pruned here
            newly_pruned_mask = (el2n_array >= prune_lo) & (el2n_array <= prune_hi) & (~pruned_mask)
        else:
            newly_pruned_mask = np.zeros_like(pruned_mask)
        newly_pruned_count = int(newly_pruned_mask.sum())
        pruned_mask = pruned_mask | newly_pruned_mask
        total_pruned = int(pruned_mask.sum())
        active_count = num_train_sequences - total_pruned

        # Write pruning table
        prune_path = os.path.join(tracking_dir, f"epoch_{display_epoch}_pruning_table.txt")
        with open(prune_path, "w") as f:
            f.write(f"Epoch {display_epoch} Pruning Table\n")
            f.write(f"Pruning enabled: {pruning_enabled}\n")
            f.write(f"Rule: prune EL2N in [{prune_lo:.4f}, {prune_hi:.4f}]  (low EL2N == easy)\n")
            f.write(f"Current active dataset size: {current_active_count}\n")
            f.write(f"Processed/tracked this epoch: {tracked_count}\n")
            f.write(f"Newly pruned this epoch: {newly_pruned_count}\n")
            f.write(f"Total pruned so far: {total_pruned}\n")
            f.write(f"Active sequences next epoch: {active_count}\n\n")
            f.write(f"{'Bucket':>20} {'% of active':>14} {'count':>14} {'action':>14}\n")
            f.write("-" * 68 + "\n")
            f.write(f"{'DID NOT TRAIN':>20} {100.0*dnt_count/max(1,current_active_count):13.2f}% {dnt_count:14,} {'KEEP':>14}\n")
            f.write(f"{'NO MASKED TOKENS':>20} {100.0*nm_count/max(1,current_active_count):13.2f}% {nm_count:14,} {'KEEP':>14}\n")

            for i in range(len(EL2N_BIN_EDGES) - 1):
                lo, hi = EL2N_BIN_EDGES[i], EL2N_BIN_EDGES[i + 1]
                count = int(((scored_vals >= lo) & (scored_vals < hi)).sum()) if scored_vals.size else 0
                if pruning_enabled and prune_lo <= lo and hi <= prune_hi:
                    action = "PRUNE CAND"
                else:
                    action = "KEEP"
                f.write(f"{f'[{lo:.1f}, {hi:.1f})':>20} "
                        f"{100.0*count/max(1,current_active_count):13.2f}% {count:14,} {action:>14}\n")

        elapsed_hours = (time.time() - train_start_time) / 3600.0
        logger.info(
            f"MLM pruning, epoch {display_epoch}: "
            f"elapsed {elapsed_hours:.2f}h, GPU-hours {elapsed_hours * world_size:.2f}, "
            f"newly pruned {newly_pruned_count:,}, total {total_pruned:,} "
            f"({100.0*total_pruned/num_train_sequences:.4f}%), "
            f"active next {active_count:,}"
        )

        # Save the cumulative pruning mask and this epoch's EL2N for every sequence
        np.save(os.path.join(tracking_dir, "pruned_mask.npy"), pruned_mask)
        np.save(os.path.join(tracking_dir, f"epoch_{display_epoch}_el2n.npy"),
                el2n_array.astype(np.float32))

        # Prepare broadcast tensor
        pruned_tensor = torch.from_numpy(pruned_mask.view(np.uint8)).to(device)
        rank0_el2n_array = el2n_array
    else:
        pruned_tensor = torch.zeros(num_train_sequences, dtype=torch.uint8, device=device)
        rank0_el2n_array = None

    # Broadcast updated mask to all ranks
    if is_distributed:
        dist.broadcast(pruned_tensor, src=0)

    pruned_mask = pruned_tensor.cpu().numpy().view(bool).copy()

    if is_distributed:
        dist.barrier()

    return pruned_mask, rank0_el2n_array


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

    train_collator = MLMTrackingDataCollator(
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
    mask_token_id = tokenizer.mask_token_id

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

    # Output folders and pruning state
    output_dir = cfg["output_dir"]
    tracking_dir = os.path.join(output_dir, "mlm_tracking")
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        os.makedirs(tracking_dir, exist_ok=True)

    pruned_mask = np.zeros(num_train_sequences, dtype=bool)
    global_step = 0
    current_epoch = 0
    log_history = []

    # Resume from checkpoint
    ckpt_dir = find_latest_checkpoint(output_dir)
    if ckpt_dir is not None:
        global_step, current_epoch, pruned_mask, log_history = load_checkpoint(
            ckpt_dir, model, optimizer, scheduler, device,
        )
        if rank == 0:
            logger.info(
                f"Resuming: step {global_step}, epoch {current_epoch}, "
                f"pruned {int(pruned_mask.sum()):,}/{num_train_sequences:,}"
            )
    elif os.path.exists(os.path.join(tracking_dir, "pruned_mask.npy")):
        # No full checkpoint but pruning mask exists (edge case)
        pruned_mask = np.load(os.path.join(tracking_dir, "pruned_mask.npy"))
        if rank == 0:
            logger.info(f"Loaded existing pruned_mask: {int(pruned_mask.sum()):,} pruned")

    # Broadcast pruned_mask to all ranks
    pm_tensor = torch.from_numpy(pruned_mask.view(np.uint8)).to(device)
    dist.broadcast(pm_tensor, src=0)
    pruned_mask = pm_tensor.cpu().numpy().view(bool).copy()

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
        logger.info(f"BF16: {use_bf16} | pruning_enabled: {cfg['pruning_enabled']} "
                    f"| signal: per-sequence EL2N | prune band "
                    f"[{cfg['pruning_lower_threshold']:.4f}, {cfg['pruning_upper_threshold']:.4f}]")

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
        # Build a fresh dataloader for this epoch
        epoch_loader, active_count = make_train_dataloader(
            train_dataset, pruned_mask, train_collator, cfg,
            rank, world_size, current_epoch,
        )
        steps_this_epoch = len(epoch_loader) // grad_accum
        if rank == 0:
            logger.info(
                f"Epoch {current_epoch}: {active_count:,} active sequences, "
                f"{steps_this_epoch:,} steps/epoch "
                f"(pruned {int(pruned_mask.sum()):,})"
            )

        # Per-epoch tracking buffers (reset every epoch)
        epoch_buffer = {}       # {line_id: (summed_EL2N, masked_count)}
        no_mask_buffer = set()  # line_ids with no masked tokens

        micro_step = 0
        for batch in epoch_loader:
            if global_step >= max_steps:
                break

            # Move to GPU
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            line_ids = batch.pop("line_id")

            # Forward pass (line_id was removed above, so the batch holds only model inputs and labels)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(**batch)

            loss = outputs.loss
            if grad_accum > 1:
                loss = loss / grad_accum
            loss.backward()

            # Per-sequence EL2N from the forward pass that already ran (no extra pass):
            # EL2N(position) = ||softmax(logits) - onehot(label)||_2
            with torch.no_grad():
                logits = outputs.logits
                labels = batch["labels"]
                input_ids = batch["input_ids"]
                bsz, seqlen = input_ids.shape[:2]

                # The same [MASK] positions that run_mlm.py scores for accuracy
                mask_pos = (input_ids == mask_token_id) & (labels != -100)

                flat_sel = mask_pos.view(-1).nonzero(as_tuple=False).flatten()
                el2n_per_seq = torch.zeros(bsz, dtype=torch.float64, device=logits.device)
                if flat_sel.numel() > 0:
                    # Select the scored positions before casting to fp32, since a full
                    # (batch, length, vocab) fp32 tensor would take hundreds of MB
                    sel_logits = logits.view(-1, logits.size(-1))[flat_sel].float()
                    sel_labels = labels.view(-1)[flat_sel]
                    err = torch.softmax(sel_logits, dim=-1)
                    err[torch.arange(err.size(0), device=err.device), sel_labels] -= 1.0
                    el2n_per_pos = err.norm(dim=-1)          # L2 over the vocab axis
                    row_idx = flat_sel // seqlen
                    el2n_per_seq.index_add_(0, row_idx, el2n_per_pos.double())

                lids_np = line_ids.cpu().numpy()
                el2n_np = el2n_per_seq.cpu().numpy()
                total_np = mask_pos.sum(dim=-1).cpu().numpy()

                for i in range(len(lids_np)):
                    lid = int(lids_np[i])
                    n_mask = int(total_np[i])
                    if n_mask == 0:
                        if lid not in epoch_buffer:
                            no_mask_buffer.add(lid)
                        continue
                    seq_el2n = float(el2n_np[i])
                    if lid in epoch_buffer:
                        prev_c, prev_n = epoch_buffer[lid]
                        epoch_buffer[lid] = (prev_c + seq_el2n, prev_n + n_mask)
                    else:
                        epoch_buffer[lid] = (seq_el2n, n_mask)
                    no_mask_buffer.discard(lid)

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
                        "active_sequences": int(num_train_sequences - pruned_mask.sum()),
                        "total_pruned": int(pruned_mask.sum()),
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
                        display_epoch, pruned_mask, epoch_buffer, no_mask_buffer,
                        num_train_sequences, max_steps, log_history,
                        output_dir, save_total_limit, tokenizer, rank,
                        train_start_time, world_size,
                    )

        # End of epoch: update the pruning mask; the next epoch's dataloader uses it
        if global_step < max_steps:
            pruned_mask, epoch_el2n_array = run_epoch_end_pruning(
                epoch_buffer, no_mask_buffer, pruned_mask,
                num_train_sequences, current_epoch,
                tracking_dir, device, train_start_time, cfg,
            )

            # Record the score of 100 randomly chosen sequences in a spreadsheet (rank 0 only)
            if rank == 0 and epoch_el2n_array is not None:
                tracked_xlsx_path = os.path.join(tracking_dir, "tracked_sequences.xlsx")
                tracked_ids_path = os.path.join(tracking_dir, "tracked_sequence_ids.npy")
                try:
                    from openpyxl import Workbook, load_workbook

                    # First epoch: pick 100 random sequences from the whole corpus.
                    # The choice depends only on the seed, not on the pruning settings,
                    # so runs with the same seed follow the same 100 sequences.
                    if not os.path.exists(tracked_ids_path):
                        all_ids = np.arange(num_train_sequences)
                        rng = np.random.RandomState(cfg["seed"])
                        n_pick = min(100, len(all_ids))
                        tracked_ids = rng.choice(all_ids, size=n_pick, replace=False)
                        tracked_ids.sort()
                        np.save(tracked_ids_path, tracked_ids)

                        wb = Workbook()
                        ws = wb.active
                        ws.title = "Tracked Sequences"
                        ws.cell(row=1, column=1, value="sequence_id")
                        for i, sid in enumerate(tracked_ids):
                            ws.cell(row=i + 2, column=1, value=int(sid))
                        wb.save(tracked_xlsx_path)
                        logger.info(f"Created tracked_sequences.xlsx with {n_pick} sequences")

                    tracked_ids = np.load(tracked_ids_path)
                    wb = load_workbook(tracked_xlsx_path)
                    ws = wb.active

                    # Find next empty column
                    col = ws.max_column + 1
                    ws.cell(row=1, column=col, value=f"epoch_{current_epoch + 1}")

                    for i, sid in enumerate(tracked_ids):
                        row = i + 2
                        # Check if already marked PRUNED in a previous column
                        prev_val = ws.cell(row=row, column=col - 1).value if col > 2 else None
                        if prev_val == "PRUNED":
                            ws.cell(row=row, column=col, value="PRUNED")
                        elif pruned_mask[sid]:
                            # Just got pruned this epoch
                            ws.cell(row=row, column=col, value="PRUNED")
                        else:
                            e = epoch_el2n_array[sid]
                            if e == SCORE_NOT_TRAINED:
                                ws.cell(row=row, column=col, value="NOT_SEEN")
                            elif e == SCORE_NO_MASK:
                                ws.cell(row=row, column=col, value="NO_MASK")
                            else:
                                ws.cell(row=row, column=col, value=round(float(e), 4))

                    wb.save(tracked_xlsx_path)
                    logger.info(f"Updated tracked_sequences.xlsx for epoch {current_epoch + 1}")
                except ImportError:
                    logger.warning("openpyxl not installed, skipping tracked_sequences.xlsx")
                except Exception as e:
                    logger.warning(f"Tracked sequences spreadsheet error (non-fatal): {e}")

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
                logger.info(f"Epoch {graceful_exit_epoch} reached, final model saved to {exit_dir}")
            dist.barrier()
            break


    total_time = (time.time() - train_start_time) / 3600.0
    if rank == 0:
        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE: {total_time:.2f}h wall-clock, "
                     f"{total_time * world_size:.2f} GPU-hours")
        logger.info("=" * 60)

        # Final save
        final_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        model.module.save_pretrained(final_dir, safe_serialization=False)
        tokenizer.save_pretrained(final_dir)
        add_model_code(final_dir)
        np.save(os.path.join(final_dir, "pruned_mask.npy"), pruned_mask)
        logger.info(f"Final model saved to: {final_dir}")

    # Final eval
    if eval_dataloader is not None:
        metrics = evaluate(model, eval_dataloader, device, use_bf16)
        if rank == 0:
            logger.info(f"FINAL EVAL: {metrics}")

    dist.destroy_process_group()


def main():
    if len(sys.argv) < 2:
        print("Usage: torchrun ... run_mlm_el2n.py <config.json>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    train(cfg)


if __name__ == "__main__":
    main()
