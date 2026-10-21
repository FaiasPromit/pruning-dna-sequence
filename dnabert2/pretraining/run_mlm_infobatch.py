#!/usr/bin/env python
"""
DNABERT-2 MLM pretraining with InfoBatch (Qin et al., 2024), used as a
dynamic-pruning baseline. The algorithm itself is in infobatch.py.

At the start of each epoch, InfoBatch picks the sequences to train on, and the
dataloader is rebuilt over that subset. Unlike Sequence Pruning, nothing is
removed permanently: every sequence can be selected again in the next epoch.

The MLM loss is recomputed here from outputs.logits so that each sequence can
carry its own weight (1 / (1 - r) for kept sequences from the below-mean
group). With all weights equal to 1 it is the model's own loss, which is
checked once at startup.

Everything else (model, optimizer, learning-rate schedule, accuracy tracking)
is the same as in run_mlm.py.

Launch with one process per GPU:
    torchrun --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> run_mlm_infobatch.py <config.json>
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

from infobatch import InfoBatch, weighted_mlm_loss, ACC_NOT_SEEN, ACC_NO_MASK

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
    cfg.setdefault("graceful_exit_epoch", None)

    # InfoBatch settings (Qin et al. defaults: r = 0.5, delta = 0.875)
    cfg.setdefault("infobatch_enabled", True)
    cfg.setdefault("infobatch_r", 0.5)
    cfg.setdefault("infobatch_delta", 0.875)
    # Annealing starts at a fraction of the total epoch budget, which is graceful_exit_epoch
    cfg.setdefault("infobatch_total_epochs", cfg.get("graceful_exit_epoch"))
    cfg.setdefault("infobatch_tracked_sequences", 100)
    cfg.setdefault("infobatch_verify_loss", True)

    # Permanent accuracy-based pruning (run_mlm.py) is never combined with InfoBatch,
    # which needs every sequence to stay selectable in every epoch
    if cfg["infobatch_enabled"] and cfg.get("pruning_enabled", False):
        logger.warning(
            "pruning_enabled is ignored by run_mlm_infobatch.py; "
            "InfoBatch never removes sequences permanently."
        )
    cfg["pruning_enabled"] = False
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


def make_train_dataloader(full_dataset, kept_indices, data_collator, cfg,
                          rank, world_size, epoch):
    """Build a fresh DataLoader over the sequences InfoBatch kept for this epoch.

    kept_indices comes from InfoBatch.build_epoch() and is the same on every
    rank because the draw is seeded from (seed, epoch).
    """
    active_indices = kept_indices.tolist()

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
                    display_epoch, infobatch,
                    num_train_sequences, max_steps, log_history,
                    output_dir, save_total_limit, tokenizer, rank,
                    train_start_time, world_size):
    """Save the model, optimizer and scheduler state, InfoBatch scores, and a run summary (rank 0 only).

    The model is saved with save_pretrained as in run_mlm.py; from_pretrained
    ignores the InfoBatch files saved next to it.
    """
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

        # InfoBatch scores and settings
        infobatch.save(ckpt_dir)

        # trainer_state.json: human-readable run summary
        ib_stats = infobatch.last_epoch_stats
        elapsed_hours = (time.time() - train_start_time) / 3600.0
        trainer_state = {
            "global_step": global_step,
            "epoch": round(display_epoch, 6),
            "epoch_integer": current_epoch,
            "max_steps": max_steps,
            "learning_rate": scheduler.get_last_lr()[0],
            "total_sequences": num_train_sequences,
            "active_sequences": ib_stats.get("n_kept", num_train_sequences),
            "skipped_this_epoch": ib_stats.get("n_dropped", 0),
            "rescaled_this_epoch": ib_stats.get("n_rescaled", 0),
            "effective_prune_pct": round(ib_stats.get("effective_prune_pct", 0.0), 4),
            "infobatch_annealing": ib_stats.get("annealing", False),
            "infobatch_score_mean": round(float(infobatch.scores.mean()), 6),
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


def load_checkpoint(ckpt_dir, model, optimizer, scheduler, device, infobatch):
    """Load model weights, optimizer, scheduler, and InfoBatch scores."""
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

    # InfoBatch scores
    infobatch.load(ckpt_dir)

    # Log history
    log_history = []
    ts_json_path = os.path.join(ckpt_dir, "trainer_state.json")
    if os.path.exists(ts_json_path):
        with open(ts_json_path) as f:
            ts_json = json.load(f)
            log_history = ts_json.get("log_history", [])

    logger.info(
        f"Resumed: step={global_step}, epoch={current_epoch}, "
        f"score_mean={infobatch.scores.mean():.4f}, "
        f"log_history={len(log_history)} entries"
    )
    return global_step, current_epoch, log_history


def _write_accuracy_dist(path, display_epoch, active_acc, active_count, tracked_count):
    """Write the accuracy distribution of this epoch's kept sequences (same format as run_mlm.py)."""
    with open(path, "w") as f:
        f.write(f"Epoch {display_epoch} Accuracy Distribution\n")
        f.write(f"Active (kept) dataset size this epoch: {active_count}\n")
        f.write(f"Processed/tracked this epoch: {tracked_count}\n\n")
        f.write(f"{'Bucket':>20} {'% of active':>14} {'count':>14}\n")
        f.write("-" * 52 + "\n")

        dnt_count = int((active_acc == ACC_NOT_SEEN).sum())
        nm_count = int((active_acc == ACC_NO_MASK).sum())
        f.write(f"{'DID NOT TRAIN':>20} {100.0*dnt_count/max(1,active_count):13.2f}% {dnt_count:14,}\n")
        f.write(f"{'NO MASKED TOKENS':>20} {100.0*nm_count/max(1,active_count):13.2f}% {nm_count:14,}\n")

        for pct in range(100, -1, -1):
            if pct == 100:
                count = int((active_acc == 1.0).sum())
            elif pct == 0:
                count = int((active_acc == 0.0).sum())
            else:
                lo, hi = pct / 100.0, (pct + 1) / 100.0
                count = int(((active_acc >= lo) & (active_acc < hi)).sum())
            f.write(f"{str(pct)+'%':>20} {100.0*count/max(1,active_count):13.2f}% {count:14,}\n")


def _write_infobatch_table(path, ep_stats, update_info, scores, elapsed_hours,
                           gpu_hours, steps_this_epoch, steps_full_epoch):
    """Write the per-epoch InfoBatch report."""
    d = ep_stats
    with open(path, "w") as f:
        f.write(f"Epoch {d['epoch_1indexed']} InfoBatch Table\n")
        f.write("=" * 70 + "\n")
        f.write(f"r (prune probability)          : {d['r']}\n")
        f.write(f"delta (anneal quantile)        : {d['delta']}\n")
        f.write(f"total epoch budget C           : {d['total_epochs']}\n")
        f.write(f"annealing starts at epoch      : {d['anneal_start_epoch_1indexed']}\n")
        f.write(f"annealing active this epoch    : {d['annealing']}\n")
        f.write(f"rescale factor 1/(1-r)         : {d['rescale_factor']:.4f}\n\n")

        f.write("SOFT PRUNING\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'score threshold (mean H_t)':<38}{d['mean_score']:>14.6f}\n")
        f.write(f"{'total sequences':<38}{d['n_total']:>14,}\n")
        f.write(f"{'below mean (prunable pool)':<38}{d['n_below_mean']:>14,}"
                f"  ({d['pct_below_mean']:.2f}%)\n")
        f.write(f"{'at/above mean (always kept)':<38}{d['n_above_mean']:>14,}\n")
        f.write(f"{'skipped this epoch':<38}{d['n_dropped']:>14,}"
                f"  ({d['effective_prune_pct']:.2f}%)\n")
        f.write(f"{'kept this epoch':<38}{d['n_kept']:>14,}\n")
        f.write(f"{'kept AND rescaled':<38}{d['n_rescaled']:>14,}\n\n")

        f.write("STEPS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'optimizer steps this epoch':<38}{steps_this_epoch:>14,}\n")
        f.write(f"{'steps if full dataset':<38}{steps_full_epoch:>14,}\n")
        saved = 100.0 * (1.0 - steps_this_epoch / max(1, steps_full_epoch))
        f.write(f"{'step saving this epoch':<38}{saved:>13.2f}%\n\n")

        f.write("SCORE UPDATE (H_{t+1}(z) = L(z) for trained z)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'sequences trained':<38}{update_info['n_sequences_trained']:>14,}\n")
        f.write(f"{'scores updated':<38}{update_info['n_scores_updated']:>14,}\n")
        f.write(f"{'never scored yet (still 1.0)':<38}{update_info['n_never_scored']:>14,}\n")
        f.write(f"{'mean score before':<38}{update_info['score_mean_before']:>14.6f}\n")
        f.write(f"{'mean score after':<38}{update_info['score_mean_after']:>14.6f}\n")
        f.write(f"{'median score':<38}{update_info['score_median']:>14.6f}\n")
        f.write(f"{'min / max score':<38}"
                f"{update_info['score_min']:>7.4f}/{update_info['score_max']:.4f}\n")
        f.write(f"{'token-mean loss over trained':<38}"
                f"{update_info['epoch_loss_mean_over_trained']:>14.6f}\n\n")

        f.write("SCORE DISTRIBUTION (loss deciles, all sequences)\n")
        f.write("-" * 70 + "\n")
        qs = np.percentile(scores, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        for q, v in zip([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], qs):
            f.write(f"{'p' + str(q):>20}{v:>20.6f}\n")
        f.write("\n")
        f.write(f"elapsed {elapsed_hours:.2f}h, GPU-hours {gpu_hours:.2f}\n")


def _append_epoch_summary_csv(path, ep_stats, update_info, steps_this_epoch,
                              steps_full_epoch, elapsed_hours, gpu_hours):
    """Append this epoch's row to the InfoBatch summary CSV."""
    header = (
        "epoch,annealing,mean_score,n_total,n_below_mean,pct_below_mean,"
        "n_skipped,n_kept,n_rescaled,effective_prune_pct,rescale_factor,"
        "steps_this_epoch,steps_full_epoch,step_saving_pct,"
        "n_trained,n_scores_updated,score_mean_after,score_median,"
        "epoch_loss_mean,elapsed_hours,gpu_hours\n"
    )
    exists = os.path.exists(path)
    with open(path, "a") as f:
        if not exists:
            f.write(header)
        d = ep_stats
        saving = 100.0 * (1.0 - steps_this_epoch / max(1, steps_full_epoch))
        f.write(
            f"{d['epoch_1indexed']},{int(d['annealing'])},{d['mean_score']:.6f},"
            f"{d['n_total']},{d['n_below_mean']},{d['pct_below_mean']:.4f},"
            f"{d['n_dropped']},{d['n_kept']},{d['n_rescaled']},"
            f"{d['effective_prune_pct']:.4f},{d['rescale_factor']:.4f},"
            f"{steps_this_epoch},{steps_full_epoch},{saving:.4f},"
            f"{update_info['n_sequences_trained']},{update_info['n_scores_updated']},"
            f"{update_info['score_mean_after']:.6f},{update_info['score_median']:.6f},"
            f"{update_info['epoch_loss_mean_over_trained']:.6f},"
            f"{elapsed_hours:.4f},{gpu_hours:.4f}\n"
        )


def run_epoch_end_infobatch(infobatch, kept_indices, ep_stats, current_epoch,
                            tracking_dir, train_start_time, world_size,
                            steps_this_epoch, steps_full_epoch, rank):
    """Sum per-sequence statistics across ranks, update the scores, and write reports.

    Returns (acc_array, scores_snapshot) on every rank; only rank 0 writes files.
    acc_array uses the same ACC_NOT_SEEN and ACC_NO_MASK markers as run_mlm.py.
    """
    acc_array, update_info = infobatch.sync_and_update()
    scores_snapshot = infobatch.scores.copy()

    if rank == 0:
        display_epoch = current_epoch + 1
        os.makedirs(tracking_dir, exist_ok=True)

        active_acc = acc_array[kept_indices]
        active_count = len(kept_indices)
        tracked_count = update_info["n_sequences_trained"]

        _write_accuracy_dist(
            os.path.join(tracking_dir, f"epoch_{display_epoch}_accuracy_dist.txt"),
            display_epoch, active_acc, active_count, tracked_count,
        )

        elapsed_hours = (time.time() - train_start_time) / 3600.0
        gpu_hours = elapsed_hours * world_size

        _write_infobatch_table(
            os.path.join(tracking_dir, f"epoch_{display_epoch}_infobatch_table.txt"),
            ep_stats, update_info, scores_snapshot,
            elapsed_hours, gpu_hours, steps_this_epoch, steps_full_epoch,
        )
        _append_epoch_summary_csv(
            os.path.join(tracking_dir, "infobatch_epoch_summary.csv"),
            ep_stats, update_info, steps_this_epoch, steps_full_epoch,
            elapsed_hours, gpu_hours,
        )
        np.save(os.path.join(tracking_dir, "infobatch_scores.npy"), scores_snapshot)

        logger.info(
            f"InfoBatch, epoch {display_epoch}: "
            f"threshold={ep_stats['mean_score']:.4f}, "
            f"below-mean {ep_stats['n_below_mean']:,} ({ep_stats['pct_below_mean']:.1f}%), "
            f"skipped {ep_stats['n_dropped']:,} ({ep_stats['effective_prune_pct']:.1f}%), "
            f"rescaled {ep_stats['n_rescaled']:,}, "
            f"steps {steps_this_epoch:,}/{steps_full_epoch:,}, "
            f"score_mean {update_info['score_mean_before']:.4f} -> "
            f"{update_info['score_mean_after']:.4f}, "
            f"elapsed {elapsed_hours:.2f}h"
        )

    infobatch.reset_epoch_stats()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return acc_array, scores_snapshot


def update_tracked_spreadsheets(tracking_dir, cfg, current_epoch, kept_indices,
                                acc_array, scores_snapshot, ep_stats,
                                rescale_mask, num_train_sequences):
    """Update two workbooks for the same 100 randomly chosen sequences.

      tracked_sequences.xlsx: per-epoch MLM accuracy (same layout as run_mlm.py)
      infobatch_tracked.xlsx: per-epoch InfoBatch state, in three sheets
            Scores:   the loss score used for the threshold decision
            Status:   SKIPPED, KEPT_RESCALED, KEPT_PLAIN, KEPT_NOT_REACHED or ANNEAL_FULL
            Accuracy: the same numbers as tracked_sequences.xlsx
    """
    try:
        from openpyxl import Workbook, load_workbook
    except ImportError:
        logger.warning("openpyxl not installed, skipping spreadsheets")
        return

    display_epoch = current_epoch + 1
    ids_path = os.path.join(tracking_dir, "tracked_sequence_ids.npy")
    acc_xlsx = os.path.join(tracking_dir, "tracked_sequences.xlsx")
    ib_xlsx = os.path.join(tracking_dir, "infobatch_tracked.xlsx")

    try:
        # First epoch: pick 100 random sequences from the whole corpus.
        # The choice depends only on the seed, so it matches run_mlm.py.
        if not os.path.exists(ids_path):
            n_pick = min(int(cfg.get("infobatch_tracked_sequences", 100)),
                         num_train_sequences)
            rng = np.random.RandomState(cfg["seed"])
            tracked_ids = rng.choice(num_train_sequences, size=n_pick, replace=False)
            tracked_ids.sort()
            np.save(ids_path, tracked_ids)

            for path, title in ((acc_xlsx, "Tracked Sequences"), (ib_xlsx, "Scores")):
                wb = Workbook()
                ws = wb.active
                ws.title = title
                ws.cell(row=1, column=1, value="sequence_id")
                for i, sid in enumerate(tracked_ids):
                    ws.cell(row=i + 2, column=1, value=int(sid))
                if path == ib_xlsx:
                    for extra in ("Status", "Accuracy"):
                        ws2 = wb.create_sheet(extra)
                        ws2.cell(row=1, column=1, value="sequence_id")
                        for i, sid in enumerate(tracked_ids):
                            ws2.cell(row=i + 2, column=1, value=int(sid))
                wb.save(path)
            logger.info(f"Created tracked spreadsheets with {n_pick} sequences")

        tracked_ids = np.load(ids_path)
        kept_set = np.zeros(num_train_sequences, dtype=bool)
        kept_set[kept_indices] = True
        annealing = ep_stats["annealing"]

        def acc_cell(sid):
            a = acc_array[sid]
            if a == ACC_NOT_SEEN:
                return "NOT_SEEN"
            if a == ACC_NO_MASK:
                return "NO_MASK"
            return round(float(a), 4)

        # Workbook 1: accuracy only
        wb = load_workbook(acc_xlsx)
        ws = wb.active
        col = ws.max_column + 1
        ws.cell(row=1, column=col, value=f"epoch_{display_epoch}")
        for i, sid in enumerate(tracked_ids):
            ws.cell(row=i + 2, column=col,
                    value="SKIPPED" if not kept_set[sid] else acc_cell(sid))
        wb.save(acc_xlsx)

        # Workbook 2: InfoBatch state
        wb = load_workbook(ib_xlsx)
        ws_scores = wb["Scores"]
        ws_status = wb["Status"]
        ws_acc = wb["Accuracy"]
        col = ws_scores.max_column + 1
        for ws in (ws_scores, ws_status, ws_acc):
            ws.cell(row=1, column=col, value=f"epoch_{display_epoch}")

        for i, sid in enumerate(tracked_ids):
            row = i + 2
            sid = int(sid)
            ws_scores.cell(row=row, column=col,
                           value=round(float(scores_snapshot[sid]), 6))
            if annealing:
                status = "ANNEAL_FULL"
            elif not kept_set[sid]:
                status = "SKIPPED"
            elif acc_array[sid] == ACC_NOT_SEEN:
                # Kept, but not processed (left out of the last incomplete batch,
                # or max_steps was reached first)
                status = "KEPT_NOT_REACHED"
            elif rescale_mask[sid]:
                status = "KEPT_RESCALED"
            else:
                status = "KEPT_PLAIN"
            ws_status.cell(row=row, column=col, value=status)
            ws_acc.cell(row=row, column=col,
                        value="SKIPPED" if not kept_set[sid] else acc_cell(sid))
        wb.save(ib_xlsx)

        logger.info(f"Updated tracked spreadsheets for epoch {display_epoch}")
    except Exception as e:
        logger.warning(f"Spreadsheet update failed (non-fatal): {e}")


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

    # Output folders and InfoBatch state
    output_dir = cfg["output_dir"]
    tracking_dir = os.path.join(output_dir, "mlm_tracking")
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        os.makedirs(tracking_dir, exist_ok=True)

    infobatch = InfoBatch(
        num_samples=num_train_sequences,
        r=cfg["infobatch_r"],
        delta=cfg["infobatch_delta"],
        total_epochs=cfg["infobatch_total_epochs"],
        seed=cfg["seed"],
        device=device,
        enabled=cfg["infobatch_enabled"],
    )

    global_step = 0
    current_epoch = 0
    log_history = []

    # Resume from checkpoint
    ckpt_dir = find_latest_checkpoint(output_dir)
    if ckpt_dir is not None:
        global_step, current_epoch, log_history = load_checkpoint(
            ckpt_dir, model, optimizer, scheduler, device, infobatch,
        )
    elif os.path.exists(os.path.join(tracking_dir, "infobatch_scores.npy")):
        # No full checkpoint but scores exist (edge case)
        infobatch.load(tracking_dir)

    # Send rank 0's scores to every rank so all ranks make the same keep/skip decisions
    sc_tensor = torch.from_numpy(infobatch.scores).to(device)
    dist.broadcast(sc_tensor, src=0)
    infobatch.scores = sc_tensor.cpu().numpy().copy()

    if rank == 0:
        logger.info(
            f"InfoBatch: r={infobatch.r}, delta={infobatch.delta}, "
            f"C={infobatch.total_epochs} epochs, "
            f"pruning runs epochs 1..{infobatch.anneal_start_epoch}, "
            f"annealing (full data) epochs "
            f"{infobatch.anneal_start_epoch + 1}..{infobatch.total_epochs}, "
            f"rescale factor {infobatch.scale:.3f}"
        )

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

    steps_full_epoch = None   # steps an epoch over the full corpus would take
    loss_verified = not cfg.get("infobatch_verify_loss", True)

    while global_step < max_steps:
        # Choose this epoch's sequences (the same on all ranks)
        kept_indices, ep_stats = infobatch.build_epoch(current_epoch)
        infobatch.reset_epoch_stats()

        epoch_loader, active_count = make_train_dataloader(
            train_dataset, kept_indices, train_collator, cfg,
            rank, world_size, current_epoch,
        )
        steps_this_epoch = len(epoch_loader) // grad_accum
        if steps_full_epoch is None or ep_stats["annealing"]:
            # Measure it from a full-corpus epoch when there is one; otherwise
            # scale up from the kept fraction
            steps_full_epoch = (
                steps_this_epoch if ep_stats["n_kept"] == ep_stats["n_total"]
                else int(round(steps_this_epoch * ep_stats["n_total"]
                               / max(1, ep_stats["n_kept"])))
            )

        if rank == 0:
            logger.info(
                f"Epoch {current_epoch + 1}/{infobatch.total_epochs}: "
                f"{active_count:,} kept "
                f"({ep_stats['n_dropped']:,} skipped, "
                f"{ep_stats['n_rescaled']:,} rescaled x{ep_stats['rescale_factor']:.2f}), "
                f"{steps_this_epoch:,} steps"
                + ("  [annealing: full corpus, no rescaling]"
                   if ep_stats["annealing"] else "")
            )

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

            # Expectation rescaling: recompute the MLM loss with per-sequence weights,
            # so kept sequences from the below-mean group count 1 / (1 - r) times
            labels = batch["labels"]
            sample_w = infobatch.batch_weights(line_ids)
            loss, per_seq_loss_sum, per_seq_loss_cnt = weighted_mlm_loss(
                outputs.logits, labels, sample_w,
            )

            # Check once that the recomputed loss with all weights 1 equals outputs.loss
            if not loss_verified:
                with torch.no_grad():
                    ref = outputs.loss.float()
                    chk, _, _ = weighted_mlm_loss(
                        outputs.logits, labels, torch.ones_like(sample_w),
                    )
                    delta = float((chk.float() - ref).abs())
                    tol = 1e-3 * max(1.0, float(ref.abs()))
                    if delta > tol:
                        raise RuntimeError(
                            f"InfoBatch loss reconstruction mismatch: "
                            f"recomputed={float(chk):.6f} vs model={float(ref):.6f} "
                            f"(delta={delta:.3e}). Stopping, because the rescaled "
                            f"loss would not match the model's loss."
                        )
                    if rank == 0:
                        logger.info(
                            f"Loss reconstruction verified "
                            f"(recomputed={float(chk):.6f}, model={float(ref):.6f}, "
                            f"delta={delta:.2e})"
                        )
                loss_verified = True

            step_loss = loss
            if grad_accum > 1:
                step_loss = step_loss / grad_accum
            step_loss.backward()

            # Per-sequence loss and MLM accuracy, summed on the GPU
            with torch.no_grad():
                preds = outputs.logits.argmax(dim=-1)
                input_ids = batch["input_ids"]
                mask_pos = (input_ids == mask_token_id) & (labels != -100)
                correct = (preds == labels) & mask_pos

                infobatch.accumulate(
                    line_ids,
                    per_seq_loss_sum,
                    per_seq_loss_cnt,
                    correct.sum(dim=-1),
                    mask_pos.sum(dim=-1),
                )

            running_loss += loss.item()
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
                        "kept_sequences": ep_stats["n_kept"],
                        "skipped_sequences": ep_stats["n_dropped"],
                        "rescaled_sequences": ep_stats["n_rescaled"],
                        "annealing": ep_stats["annealing"],
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
                        display_epoch, infobatch,
                        num_train_sequences, max_steps, log_history,
                        output_dir, save_total_limit, tokenizer, rank,
                        train_start_time, world_size,
                    )

        # End of epoch: update scores and write reports
        epoch_acc_array, scores_snapshot = run_epoch_end_infobatch(
            infobatch, kept_indices, ep_stats, current_epoch,
            tracking_dir, train_start_time, world_size,
            steps_this_epoch, steps_full_epoch, rank,
        )

        if rank == 0:
            update_tracked_spreadsheets(
                tracking_dir, cfg, current_epoch, kept_indices,
                epoch_acc_array, scores_snapshot, ep_stats,
                infobatch.last_rescale_mask, num_train_sequences,
            )

            # Projected total steps for the full epoch budget, compared with max_steps
            done_frac = (current_epoch + 1) / max(1, infobatch.total_epochs)
            projected = int(round(global_step / max(done_frac, 1e-9)))
            logger.info(
                f"Projection: {global_step:,} steps after "
                f"{current_epoch + 1}/{infobatch.total_epochs} epochs -> "
                f"~{projected:,} steps for the full budget "
                f"(max_steps={max_steps:,}, "
                f"{100.0 * projected / max(1, max_steps):.1f}% of LR schedule)"
            )

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
                infobatch.save(exit_dir)
                logger.info(
                    f"Epoch {graceful_exit_epoch} reached, final model saved "
                    f"to {exit_dir} (global_step={global_step:,}/{max_steps:,}, "
                    f"LR schedule "
                    f"{100.0 * global_step / max(1, max_steps):.1f}% complete)"
                )
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
        infobatch.save(final_dir)
        logger.info(f"Final model saved to: {final_dir}")

    # Final eval
    if eval_dataloader is not None:
        metrics = evaluate(model, eval_dataloader, device, use_bf16)
        if rank == 0:
            logger.info(f"FINAL EVAL: {metrics}")

    dist.destroy_process_group()


def main():
    if len(sys.argv) < 2:
        print("Usage: torchrun ... run_mlm_infobatch.py <config.json>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    train(cfg)


if __name__ == "__main__":
    main()
