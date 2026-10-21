#!/usr/bin/env python
"""
DNABERT-2 masked language model (MLM) pretraining with Sequence Pruning (SP).

During each epoch, the script records each sequence's MLM accuracy from the
training forward pass, counting only the [MASK] positions. At the end of the
epoch, every active sequence whose accuracy lies in
[pruning_lower_threshold, pruning_upper_threshold] is removed from the corpus
for the rest of the run, and the next epoch's dataloader is built over the
remaining sequences only. The learning-rate schedule stays tied to max_steps,
and the run stops at max_steps or at graceful_exit_epoch, whichever comes first.

Optional config settings:
  pruning_window          Average each sequence's accuracy over this many recent
                          epochs before applying the thresholds. The default, 1,
                          is the plain SP rule.
  random_subset_fraction  Keep only this fraction of the corpus, chosen at random
                          before training (random pruning, RP). Use it together
                          with "pruning_enabled": false.

Launch with one process per GPU:
    torchrun --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> run_mlm.py <config.json>
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
    cfg.setdefault("pruning_enabled", True)
    cfg.setdefault("pruning_lower_threshold", 0.80)
    cfg.setdefault("pruning_upper_threshold", 1.00)
    cfg.setdefault("pruning_window", 1)
    cfg.setdefault("random_subset_fraction", None)
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
                    train_start_time, world_size, acc_history):
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
        np.save(os.path.join(ckpt_dir, "acc_history.npy"), acc_history)
        with open(os.path.join(ckpt_dir, "pruning_buffer.json"), "w") as f:
            json.dump({
                "epoch_buffer": {str(k): list(v) for k, v in epoch_buffer.items()},
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


def load_checkpoint(ckpt_dir, model, optimizer, scheduler, device, pruning_window):
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

    # Accuracy history for the averaging window. Start a fresh one if the
    # checkpoint has none or if pruning_window has changed.
    acc_history = None
    ah_path = os.path.join(ckpt_dir, "acc_history.npy")
    if os.path.exists(ah_path):
        loaded = np.load(ah_path)
        if loaded.shape == (len(pruned_mask), pruning_window):
            acc_history = loaded
        else:
            logger.warning(
                f"acc_history shape {loaded.shape} != "
                f"{(len(pruned_mask), pruning_window)}; reinitializing history"
            )
    if acc_history is None:
        acc_history = np.full((len(pruned_mask), pruning_window), -1.0, dtype=np.float32)

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
    return global_step, current_epoch, pruned_mask, log_history, acc_history


def run_epoch_end_pruning(epoch_buffer, no_mask_buffer, pruned_mask,
                          num_train_sequences, current_epoch,
                          tracking_dir, device, train_start_time, cfg,
                          acc_history):
    """Gather per-sequence accuracy from all ranks and update the pruning mask.

    Each sequence's accuracy is averaged over its last `pruning_window` epochs,
    and sequences whose average lies in [pruning_lower_threshold,
    pruning_upper_threshold] are marked as pruned. A sequence is eligible only if
    it has a real accuracy in each of those epochs. In the accuracy arrays, -1
    means the sequence was not processed in an epoch and -2 means it had no
    [MASK] positions. With pruning_window = 1 this is the plain SP rule.

    acc_history: rank-0 array of shape (num_train_sequences, pruning_window)
                 holding each sequence's accuracy in its most recent epochs
                 (-1 for an epoch without a real accuracy). Unused on other ranks.

    Returns (updated_pruned_mask, acc_array, avg_array, acc_history), where
    acc_array is this epoch's accuracy and avg_array is the windowed average
    used for the pruning decision (both None on ranks other than 0).
    """
    is_distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_distributed else 1
    rank = dist.get_rank() if is_distributed else 0

    # Pack local tracking data into tensors
    local_lids = list(epoch_buffer.keys())
    local_correct = [epoch_buffer[k][0] for k in local_lids]
    local_total = [epoch_buffer[k][1] for k in local_lids]

    t_lids = torch.tensor(local_lids or [0], dtype=torch.long, device=device)
    t_correct = torch.tensor(local_correct or [0], dtype=torch.float32, device=device)
    t_total = torch.tensor(local_total or [0], dtype=torch.float32, device=device)
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
        t_correct_padded = torch.zeros(max_size, dtype=torch.float32, device=device)
        t_total_padded = torch.zeros(max_size, dtype=torch.float32, device=device)
        nm_lids_padded = torch.zeros(nm_max_size, dtype=torch.long, device=device)

        t_lids_padded[:actual_lid_count] = t_lids[:actual_lid_count]
        t_correct_padded[:actual_lid_count] = t_correct[:actual_lid_count]
        t_total_padded[:actual_lid_count] = t_total[:actual_lid_count]
        nm_lids_padded[:actual_nm_count] = nm_lids[:actual_nm_count]

        all_lids = [torch.zeros_like(t_lids_padded) for _ in range(world_size)]
        all_correct = [torch.zeros_like(t_correct_padded) for _ in range(world_size)]
        all_total = [torch.zeros_like(t_total_padded) for _ in range(world_size)]
        nm_all_lids = [torch.zeros_like(nm_lids_padded) for _ in range(world_size)]

        dist.all_gather(all_lids, t_lids_padded)
        dist.all_gather(all_correct, t_correct_padded)
        dist.all_gather(all_total, t_total_padded)
        dist.all_gather(nm_all_lids, nm_lids_padded)
    else:
        all_sizes = [local_size]
        nm_all_sizes = [nm_local_size]
        all_lids = [t_lids]
        all_correct = [t_correct]
        all_total = [t_total]
        nm_all_lids = [nm_lids]

    # Merge on rank 0 and update pruning mask
    if rank == 0:
        display_epoch = current_epoch + 1  # 1-indexed for consistency with trainer_state
        merged = {}
        for r in range(world_size):
            sz = all_sizes[r].item()
            r_lids = all_lids[r][:sz].cpu().numpy()
            r_correct = all_correct[r][:sz].cpu().numpy()
            r_total = all_total[r][:sz].cpu().numpy()
            for lid, c, t in zip(r_lids, r_correct, r_total):
                lid = int(lid)
                if lid in merged:
                    merged[lid] = (merged[lid][0] + c, merged[lid][1] + t)
                else:
                    merged[lid] = (c, t)

        acc_array = np.full(num_train_sequences, -1.0, dtype=np.float32)
        for lid, (c, t) in merged.items():
            acc_array[lid] = -2.0 if t == 0 else c / t

        no_mask_set = set()
        for r in range(world_size):
            sz = nm_all_sizes[r].item()
            for lid in nm_all_lids[r][:sz].cpu().numpy():
                no_mask_set.add(int(lid))
        for lid in no_mask_set:
            if acc_array[lid] == -1.0:
                acc_array[lid] = -2.0

        # Store this epoch's accuracy in the history. Each of the `window` slots
        # holds the accuracy from one recent epoch, and -1.0 marks an epoch
        # without a real accuracy (not processed, or no masked tokens).
        window = int(cfg.get("pruning_window", 1))
        slot = current_epoch % window
        valid_this_epoch = acc_array >= 0.0
        acc_history[:, slot] = np.where(valid_this_epoch, acc_array, -1.0)

        # Windowed average. A sequence is eligible for pruning only if every slot
        # holds a real accuracy; other sequences get avg = -1.0 and are skipped.
        eligible = (acc_history >= 0.0).all(axis=1)
        avg_array = np.full(num_train_sequences, -1.0, dtype=np.float32)
        avg_array[eligible] = acc_history[eligible].mean(axis=1)

        # Stats on currently-active sequences
        keep_before = ~pruned_mask
        current_line_ids = np.where(keep_before)[0]
        current_active_count = len(current_line_ids)
        current_active_acc = acc_array[current_line_ids]
        tracked_count = len(merged) + int((acc_array == -2.0).sum())

        # Windowed average of the currently active sequences, which is what the
        # pruning decision and the pruning table use
        current_active_avg = avg_array[current_line_ids]

        os.makedirs(tracking_dir, exist_ok=True)

        # Write accuracy distribution
        stats_path = os.path.join(tracking_dir, f"epoch_{display_epoch}_accuracy_dist.txt")
        with open(stats_path, "w") as f:
            f.write(f"Epoch {display_epoch} Accuracy Distribution\n")
            f.write(f"Current active dataset size: {current_active_count}\n")
            f.write(f"Processed/tracked this epoch: {tracked_count}\n\n")
            f.write(f"{'Bucket':>20} {'% of active':>14} {'count':>14}\n")
            f.write("-" * 52 + "\n")

            dnt_count = int((current_active_acc == -1.0).sum())
            nm_count = int((current_active_acc == -2.0).sum())

            f.write(f"{'DID NOT TRAIN':>20} {100.0*dnt_count/max(1,current_active_count):13.2f}% {dnt_count:14,}\n")
            f.write(f"{'NO MASKED TOKENS':>20} {100.0*nm_count/max(1,current_active_count):13.2f}% {nm_count:14,}\n")

            for pct in range(100, -1, -1):
                if pct == 100:
                    count = int((current_active_acc == 1.0).sum())
                elif pct == 0:
                    count = int((current_active_acc == 0.0).sum())
                else:
                    lo, hi = pct / 100.0, (pct + 1) / 100.0
                    count = int(((current_active_acc >= lo) & (current_active_acc < hi)).sum())
                pct_active = 100.0 * count / max(1, current_active_count)
                f.write(f"{str(pct)+'%':>20} {pct_active:13.2f}% {count:14,}\n")

        # Prune sequences whose windowed average lies in the threshold range.
        # Ineligible sequences have avg_array = -1.0, so they are never pruned here.
        pruning_enabled = cfg.get("pruning_enabled", True)
        prune_lo = cfg.get("pruning_lower_threshold", 0.80)
        prune_hi = cfg.get("pruning_upper_threshold", 1.00)

        if pruning_enabled:
            newly_pruned_mask = (avg_array >= prune_lo) & (avg_array <= prune_hi) & (~pruned_mask)
        else:
            newly_pruned_mask = np.zeros_like(pruned_mask)
        newly_pruned_count = int(newly_pruned_mask.sum())
        pruned_mask = pruned_mask | newly_pruned_mask
        total_pruned = int(pruned_mask.sum())
        active_count = num_train_sequences - total_pruned

        # Write the pruning table, built from the same windowed average as the
        # pruning decision; NOT ELIGIBLE counts sequences with avg == -1.0
        ineligible_count = int((current_active_avg == -1.0).sum())
        prune_path = os.path.join(tracking_dir, f"epoch_{display_epoch}_pruning_table.txt")
        with open(prune_path, "w") as f:
            f.write(f"Epoch {display_epoch} Pruning Table\n")
            f.write(f"Pruning enabled: {pruning_enabled}\n")
            f.write(f"Pruning threshold (avg over last {window} epochs): [{prune_lo:.2f}, {prune_hi:.2f}]\n")
            f.write(f"Current active dataset size: {current_active_count}\n")
            f.write(f"Processed/tracked this epoch: {tracked_count}\n")
            f.write(f"Newly pruned this epoch: {newly_pruned_count}\n")
            f.write(f"Total pruned so far: {total_pruned}\n")
            f.write(f"Active sequences next epoch: {active_count}\n\n")
            f.write(f"Buckets below are over the {window}-epoch average accuracy.\n")
            f.write(f"{'Bucket':>20} {'% of active':>14} {'count':>14} {'action':>14}\n")
            f.write("-" * 68 + "\n")
            f.write(f"{'NOT ELIGIBLE':>20} {100.0*ineligible_count/max(1,current_active_count):13.2f}% {ineligible_count:14,} {'KEEP':>14}\n")

            for pct in range(100, -1, -1):
                if pct == 100:
                    count = int((current_active_avg == 1.0).sum())
                elif pct == 0:
                    count = int((current_active_avg == 0.0).sum())
                else:
                    lo, hi = pct / 100.0, (pct + 1) / 100.0
                    count = int(((current_active_avg >= lo) & (current_active_avg < hi)).sum())
                frac = pct / 100.0
                if pruning_enabled and prune_lo <= frac <= prune_hi:
                    action = "PRUNE CAND"
                else:
                    action = "KEEP"
                pct_active = 100.0 * count / max(1, current_active_count)
                f.write(f"{str(pct)+'%':>20} {pct_active:13.2f}% {count:14,} {action:>14}\n")

        elapsed_hours = (time.time() - train_start_time) / 3600.0
        logger.info(
            f"MLM pruning, epoch {display_epoch}: "
            f"elapsed {elapsed_hours:.2f}h, GPU-hours {elapsed_hours * world_size:.2f}, "
            f"newly pruned {newly_pruned_count:,}, total {total_pruned:,} "
            f"({100.0*total_pruned/num_train_sequences:.4f}%), "
            f"active next {active_count:,}"
        )

        # Save the cumulative pruning mask
        np.save(os.path.join(tracking_dir, "pruned_mask.npy"), pruned_mask)

        # Prepare broadcast tensor
        pruned_tensor = torch.from_numpy(pruned_mask.view(np.uint8)).to(device)
        rank0_acc_array = acc_array
        rank0_avg_array = avg_array
    else:
        pruned_tensor = torch.zeros(num_train_sequences, dtype=torch.uint8, device=device)
        rank0_acc_array = None
        rank0_avg_array = None

    # Broadcast updated mask to all ranks
    if is_distributed:
        dist.broadcast(pruned_tensor, src=0)

    pruned_mask = pruned_tensor.cpu().numpy().view(bool).copy()

    if is_distributed:
        dist.barrier()

    return pruned_mask, rank0_acc_array, rank0_avg_array, acc_history


def train(cfg):
    # Distributed setup. The long timeout covers the wait while rank 0 tokenizes the corpus.
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

    # Per-sequence accuracy over the last `pruning_window` epochs (used on rank 0)
    pruning_window = int(cfg.get("pruning_window", 1))
    acc_history = np.full((num_train_sequences, pruning_window), -1.0, dtype=np.float32)

    # Random pruning (RP): keep a random subset of the corpus, chosen once before
    # training, by marking every other sequence as removed. Sequence IDs stay the
    # same as in all other runs, and the subset depends only on the seed.
    if cfg.get("random_subset_fraction") is not None:
        frac = float(cfg["random_subset_fraction"])
        n_keep = int(round(frac * num_train_sequences))
        keep_ids = np.random.default_rng(cfg["seed"]).permutation(num_train_sequences)[:n_keep]
        pruned_mask[:] = True
        pruned_mask[keep_ids] = False
        if rank == 0:
            logger.info(
                f"Random pruning: keeping {n_keep:,}/{num_train_sequences:,} sequences "
                f"({100.0 * n_keep / num_train_sequences:.4f}%), seed={cfg['seed']}"
            )

    # Resume from checkpoint
    ckpt_dir = find_latest_checkpoint(output_dir)
    if ckpt_dir is not None:
        global_step, current_epoch, pruned_mask, log_history, acc_history = load_checkpoint(
            ckpt_dir, model, optimizer, scheduler, device, pruning_window,
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
        epoch_buffer = {}       # {line_id: (correct_count, masked_count)}
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

            # Per-sequence MLM accuracy, counted on [MASK] positions only
            with torch.no_grad():
                logits = outputs.logits
                preds = logits.argmax(dim=-1)
                labels = batch["labels"]
                input_ids = batch["input_ids"]

                mask_pos = (input_ids == mask_token_id) & (labels != -100)
                correct = (preds == labels) & mask_pos

                lids_np = line_ids.cpu().numpy()
                correct_np = correct.sum(dim=-1).cpu().numpy()
                total_np = mask_pos.sum(dim=-1).cpu().numpy()

                for i in range(len(lids_np)):
                    lid = int(lids_np[i])
                    n_mask = int(total_np[i])
                    if n_mask == 0:
                        if lid not in epoch_buffer:
                            no_mask_buffer.add(lid)
                        continue
                    n_correct = int(correct_np[i])
                    if lid in epoch_buffer:
                        prev_c, prev_n = epoch_buffer[lid]
                        epoch_buffer[lid] = (prev_c + n_correct, prev_n + n_mask)
                    else:
                        epoch_buffer[lid] = (n_correct, n_mask)
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
                        train_start_time, world_size, acc_history,
                    )

        # End of epoch: update the pruning mask; the next epoch's dataloader uses it
        if global_step < max_steps:
            pruned_before_mask = pruned_mask.copy()
            pruned_mask, epoch_acc_array, epoch_avg_array, acc_history = run_epoch_end_pruning(
                epoch_buffer, no_mask_buffer, pruned_mask,
                num_train_sequences, current_epoch,
                tracking_dir, device, train_start_time, cfg,
                acc_history,
            )

            # Record the accuracy of 100 randomly chosen sequences in spreadsheets (rank 0 only)
            if rank == 0 and epoch_acc_array is not None:
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
                        if pruned_before_mask[sid]:
                            # Pruned in an earlier epoch, so not trained this epoch
                            ws.cell(row=row, column=col, value="PRUNED")
                        else:
                            # Trained this epoch (even if pruned at its end): record its accuracy
                            acc = epoch_acc_array[sid]
                            if acc == -1.0:
                                ws.cell(row=row, column=col, value="NOT_SEEN")
                            elif acc == -2.0:
                                ws.cell(row=row, column=col, value="NO_MASK")
                            else:
                                ws.cell(row=row, column=col, value=round(float(acc), 4))

                    wb.save(tracked_xlsx_path)
                    logger.info(f"Updated tracked_sequences.xlsx for epoch {current_epoch + 1}")

                    # Second spreadsheet: windowed-average accuracy of the same sequences
                    avg_xlsx_path = os.path.join(tracking_dir, "tracked_sequences_avg.xlsx")
                    if not os.path.exists(avg_xlsx_path):
                        wb_a = Workbook()
                        ws_a = wb_a.active
                        ws_a.title = "Tracked Avg"
                        ws_a.cell(row=1, column=1, value="sequence_id")
                        for i, sid in enumerate(tracked_ids):
                            ws_a.cell(row=i + 2, column=1, value=int(sid))
                        wb_a.save(avg_xlsx_path)

                    wb_a = load_workbook(avg_xlsx_path)
                    ws_a = wb_a.active
                    col_a = ws_a.max_column + 1
                    ws_a.cell(row=1, column=col_a, value=f"epoch_{current_epoch + 1}")
                    for i, sid in enumerate(tracked_ids):
                        row = i + 2
                        if pruned_before_mask[sid]:
                            ws_a.cell(row=row, column=col_a, value="PRUNED")
                        else:
                            avg = epoch_avg_array[sid]
                            if avg < 0.0:
                                # No real accuracy in some epoch of the window, so not eligible
                                ws_a.cell(row=row, column=col_a, value="NOT_ENOUGH")
                            else:
                                ws_a.cell(row=row, column=col_a, value=round(float(avg), 4))
                    wb_a.save(avg_xlsx_path)
                    logger.info(f"Updated tracked_sequences_avg.xlsx for epoch {current_epoch + 1}")
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
        print("Usage: torchrun ... run_mlm.py <config.json>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    train(cfg)


if __name__ == "__main__":
    main()
