#!/usr/bin/env python
"""
HyenaDNA CLM pretraining with perplexity as the pruning score (the Perp_* ablation).

The same as run_clm.py, except for the score used to prune. A sequence's
perplexity in an epoch is exp(mean negative log-likelihood) over its predicted
positions, computed from the training forward pass (Marion et al., 2023). It is
at least 1, and low perplexity means the model already predicts the sequence
well. At the end of each epoch, every active sequence with
perplexity <= pruning_perplexity_threshold is removed for the rest of the run.
The lower bound of the pruning range is therefore always 1 and only the upper
bound changes between variants. As with accuracy in run_clm.py, the number of
sequences removed per epoch is not fixed; it is however many reached the bound.

Launch with one process per GPU:
    torchrun --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> run_clm_perplexity.py <config.json>
"""

import json
import logging
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset
from transformers import set_seed
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# The HyenaDNA model code and tokenizer live in ../model_code
MODEL_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_code")

# Special values in the per-sequence score array. A real perplexity is at least
# 1, so these negative values never clash with one and are left out by a > 0 test.
SCORE_NOT_TRAINED = -1.0   # sequence was active but never drawn this epoch
SCORE_NO_PRED = -2.0       # sequence had zero predicted positions


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault("gradient_accumulation_steps", 1)
    cfg.setdefault("max_grad_norm", 1.0)
    cfg.setdefault("dataloader_num_workers", 4)
    cfg.setdefault("dataloader_pin_memory", True)
    cfg.setdefault("dataloader_drop_last", True)
    cfg.setdefault("preprocessing_num_workers", 16)
    cfg.setdefault("save_steps", 5000)
    cfg.setdefault("eval_steps", 5000)
    cfg.setdefault("save_total_limit", 10)
    cfg.setdefault("logging_steps", 100)
    cfg.setdefault("seed", 42)
    cfg.setdefault("bf16", True)
    cfg.setdefault("max_seq_length", 1024)
    cfg.setdefault("lr_scheduler_type", "cosine")
    cfg.setdefault("warmup_steps", 0)              # learning-rate warmup (there is no sequence-length warmup)
    cfg.setdefault("weight_decay", 0.1)
    cfg.setdefault("adam_beta1", 0.9)
    cfg.setdefault("adam_beta2", 0.999)
    cfg.setdefault("adam_epsilon", 1e-8)
    cfg.setdefault("add_bos", False)               # by default the input is the nucleotides only
    cfg.setdefault("add_eos", False)
    # Perplexity pruning (off unless pruning_enabled is true)
    cfg.setdefault("pruning_enabled", False)
    cfg.setdefault("pruning_perplexity_threshold", 3.45)  # prune sequences with perplexity <= this
    cfg.setdefault("graceful_exit_epoch", None)
    cfg.setdefault("tracking_dir_name", "clm_tracking")
    return cfg


class CLMTrackingDataCollator:
    """Pads input_ids, builds the labels and keeps each sequence's line_id.

    The labels are a copy of input_ids with padding set to -100; the shift by one
    position happens inside HyenaDNAForCausalLM. All windows have the same length,
    so in practice nothing is padded.
    """

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features, return_tensors=None):
        line_ids = [int(f.get("line_id", -1)) for f in features]
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
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "line_id": torch.tensor(line_ids, dtype=torch.long),
        }


def prepare_datasets(cfg, tokenizer, rank, world_size):
    """Load the text file (one window per line), tokenize it, and give each training sequence a line_id.

    A sequence's line_id is its position among the non-empty lines of train_file,
    counting from 0. It depends only on the training file, never on the pruning
    settings, so the same line_id refers to the same sequence in every run on that
    corpus, and all saved per-sequence arrays are indexed by it.
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

    def add_line_ids(example, idx):
        return {"line_id": idx}

    num_proc = cfg["preprocessing_num_workers"]

    def _build():
        raw = load_dataset("text", data_files=data_files)
        tok = raw.map(tokenize_fn, batched=True, num_proc=num_proc,
                      remove_columns=["text"], load_from_cache_file=True, desc="Tokenizing")
        if "train" in tok:
            tok["train"] = tok["train"].map(
                add_line_ids, with_indices=True, num_proc=num_proc,
                load_from_cache_file=True, desc="Adding line IDs")
        return tok

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


def make_train_dataloader(full_dataset, pruned_mask, data_collator, cfg, rank, world_size, epoch):
    active_indices = np.where(~pruned_mask)[0].tolist()
    subset = Subset(full_dataset, active_indices)
    sampler = DistributedSampler(subset, num_replicas=world_size, rank=rank,
                                 shuffle=True, drop_last=True, seed=cfg["seed"])
    sampler.set_epoch(epoch)
    loader = DataLoader(
        subset, batch_size=cfg["per_device_train_batch_size"], sampler=sampler,
        collate_fn=data_collator, num_workers=cfg["dataloader_num_workers"],
        pin_memory=cfg["dataloader_pin_memory"], drop_last=cfg["dataloader_drop_last"],
    )
    return loader, len(active_indices)


def make_eval_dataloader(eval_dataset, data_collator, cfg, rank, world_size):
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
        batch.pop("line_id", None)
        labels = batch["labels"]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(**batch)
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


def save_checkpoint(model, optimizer, scheduler, global_step, current_epoch, display_epoch,
                    pruned_mask, epoch_buffer, no_pred_buffer, num_train_sequences, max_steps,
                    log_history, output_dir, save_total_limit, tokenizer, rank,
                    train_start_time, world_size):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        model.module.save_pretrained(ckpt_dir, safe_serialization=False)
        tokenizer.save_pretrained(ckpt_dir)

        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "current_epoch": current_epoch,
        }, os.path.join(ckpt_dir, "training_state.pt"))

        np.save(os.path.join(ckpt_dir, "pruned_mask.npy"), pruned_mask)
        with open(os.path.join(ckpt_dir, "pruning_buffer.json"), "w") as f:
            json.dump({
                "epoch_buffer": {str(k): [float(v[0]), int(v[1])] for k, v in epoch_buffer.items()},
                "no_pred_buffer": list(no_pred_buffer),
            }, f)

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
    raise FileNotFoundError(f"No model weights (pytorch_model.bin / model.safetensors) in {ckpt_dir}")


def load_checkpoint(ckpt_dir, model, optimizer, scheduler, device):
    logger.info(f"Resuming from checkpoint: {ckpt_dir}")
    state_dict = _load_state_dict(ckpt_dir, device)
    model.module.load_state_dict(state_dict)

    ts = torch.load(os.path.join(ckpt_dir, "training_state.pt"), map_location=device)
    optimizer.load_state_dict(ts["optimizer"])
    scheduler.load_state_dict(ts["scheduler"])
    global_step = ts["global_step"]
    current_epoch = ts["current_epoch"]

    pruned_mask = np.load(os.path.join(ckpt_dir, "pruned_mask.npy"))

    log_history = []
    ts_json_path = os.path.join(ckpt_dir, "trainer_state.json")
    if os.path.exists(ts_json_path):
        with open(ts_json_path) as f:
            log_history = json.load(f).get("log_history", [])

    logger.info(f"Resumed: step={global_step}, epoch={current_epoch}, "
                f"pruned={int(pruned_mask.sum()):,}, log_history={len(log_history)} entries")
    return global_step, current_epoch, pruned_mask, log_history


# Perplexity bins for the per-epoch histogram. With an alphabet of 4 to 5
# characters, perplexity on DNA lies roughly between 1 and 12; the last bin
# takes anything above.
PPL_BIN_EDGES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0,
                 5.0, 6.0, 8.0, 12.0, float("inf")]


def run_epoch_end_pruning(epoch_buffer, no_pred_buffer, pruned_mask, num_train_sequences,
                          current_epoch, tracking_dir, device, train_start_time, cfg):
    """Gather each sequence's perplexity from all ranks and prune.

    epoch_buffer holds {line_id: (summed NLL, summed number of predicted tokens)}.
    Summing both separately gives the right value when a sequence is seen in
    several pieces: perplexity = exp(sum NLL / sum tokens).

    Every active sequence with perplexity <= pruning_perplexity_threshold is
    removed. Rank 0 writes the perplexity histogram and the pruning table of the
    epoch. With "pruning_enabled": false nothing is pruned and the mask is
    returned unchanged.

    Returns (updated pruned_mask, perplexity array on rank 0 and None elsewhere).
    """
    is_distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_distributed else 1
    rank = dist.get_rank() if is_distributed else 0

    local_lids = list(epoch_buffer.keys())
    local_nll = [epoch_buffer[k][0] for k in local_lids]
    local_total = [epoch_buffer[k][1] for k in local_lids]

    t_lids = torch.tensor(local_lids or [0], dtype=torch.long, device=device)
    t_nll = torch.tensor(local_nll or [0.0], dtype=torch.float64, device=device)
    t_total = torch.tensor(local_total or [0], dtype=torch.float64, device=device)
    nm_lids = torch.tensor(list(no_pred_buffer) or [0], dtype=torch.long, device=device)

    actual_lid_count = len(local_lids)
    actual_nm_count = len(no_pred_buffer)
    local_size = torch.tensor([actual_lid_count], dtype=torch.long, device=device)
    nm_local_size = torch.tensor([actual_nm_count], dtype=torch.long, device=device)

    if is_distributed:
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        nm_all_sizes = [torch.zeros_like(nm_local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        dist.all_gather(nm_all_sizes, nm_local_size)

        max_size = max(max((s.item() for s in all_sizes), default=1), 1)
        nm_max_size = max(max((s.item() for s in nm_all_sizes), default=1), 1)

        t_lids_padded = torch.zeros(max_size, dtype=torch.long, device=device)
        t_nll_padded = torch.zeros(max_size, dtype=torch.float64, device=device)
        t_total_padded = torch.zeros(max_size, dtype=torch.float64, device=device)
        nm_lids_padded = torch.zeros(nm_max_size, dtype=torch.long, device=device)

        t_lids_padded[:actual_lid_count] = t_lids[:actual_lid_count]
        t_nll_padded[:actual_lid_count] = t_nll[:actual_lid_count]
        t_total_padded[:actual_lid_count] = t_total[:actual_lid_count]
        nm_lids_padded[:actual_nm_count] = nm_lids[:actual_nm_count]

        all_lids = [torch.zeros_like(t_lids_padded) for _ in range(world_size)]
        all_nll = [torch.zeros_like(t_nll_padded) for _ in range(world_size)]
        all_total = [torch.zeros_like(t_total_padded) for _ in range(world_size)]
        nm_all_lids = [torch.zeros_like(nm_lids_padded) for _ in range(world_size)]

        dist.all_gather(all_lids, t_lids_padded)
        dist.all_gather(all_nll, t_nll_padded)
        dist.all_gather(all_total, t_total_padded)
        dist.all_gather(nm_all_lids, nm_lids_padded)
    else:
        all_sizes = [local_size]
        nm_all_sizes = [nm_local_size]
        all_lids = [t_lids]
        all_nll = [t_nll]
        all_total = [t_total]
        nm_all_lids = [nm_lids]

    if rank == 0:
        display_epoch = current_epoch + 1

        # merge per-rank (summed NLL, summed tokens) for each line_id
        merged = {}
        for r in range(world_size):
            sz = all_sizes[r].item()
            r_lids = all_lids[r][:sz].cpu().numpy()
            r_nll = all_nll[r][:sz].cpu().numpy()
            r_total = all_total[r][:sz].cpu().numpy()
            for lid, s, t in zip(r_lids, r_nll, r_total):
                lid = int(lid)
                if lid in merged:
                    merged[lid] = (merged[lid][0] + s, merged[lid][1] + t)
                else:
                    merged[lid] = (s, t)

        ppl_array = np.full(num_train_sequences, SCORE_NOT_TRAINED, dtype=np.float64)
        for lid, (s, t) in merged.items():
            if t <= 0:
                ppl_array[lid] = SCORE_NO_PRED
            else:
                mean_nll = float(s) / float(t)
                # clamp before exp so a pathological sequence cannot overflow
                ppl_array[lid] = float(np.exp(min(mean_nll, 50.0)))

        no_pred_set = set()
        for r in range(world_size):
            sz = nm_all_sizes[r].item()
            for lid in nm_all_lids[r][:sz].cpu().numpy():
                no_pred_set.add(int(lid))
        for lid in no_pred_set:
            if ppl_array[lid] == SCORE_NOT_TRAINED:
                ppl_array[lid] = SCORE_NO_PRED

        # pools: active vs. active-and-scored
        active_idx = np.where(~pruned_mask)[0]
        current_active_count = active_idx.size

        # Only sequences with a real perplexity this epoch can be pruned
        scored_mask = (~pruned_mask) & (ppl_array > 0.0)
        scored_idx = np.where(scored_mask)[0]
        scored_vals = ppl_array[scored_idx]

        tracked_count = len(merged) + len(no_pred_set)

        os.makedirs(tracking_dir, exist_ok=True)

        # per-epoch perplexity distribution
        active_scores = ppl_array[active_idx]
        dnt_count = int((active_scores == SCORE_NOT_TRAINED).sum())
        nm_count = int((active_scores == SCORE_NO_PRED).sum())

        stats_path = os.path.join(tracking_dir, f"epoch_{display_epoch}_perplexity_dist.txt")
        with open(stats_path, "w") as f:
            f.write(f"Epoch {display_epoch} Per-Sequence Perplexity Distribution\n")
            f.write(f"Current active dataset size: {current_active_count}\n")
            f.write(f"Processed/tracked this epoch: {tracked_count}\n")
            f.write(f"Scored (rankable) active sequences: {scored_idx.size}\n\n")

            f.write(f"{'Bucket':>20} {'% of active':>14} {'count':>14}\n")
            f.write("-" * 52 + "\n")
            f.write(f"{'DID NOT TRAIN':>20} {100.0*dnt_count/max(1,current_active_count):13.2f}% {dnt_count:14,}\n")
            f.write(f"{'NO PRED TOKENS':>20} {100.0*nm_count/max(1,current_active_count):13.2f}% {nm_count:14,}\n")

            if scored_vals.size:
                for i in range(len(PPL_BIN_EDGES) - 1):
                    lo, hi = PPL_BIN_EDGES[i], PPL_BIN_EDGES[i + 1]
                    count = int(((scored_vals >= lo) & (scored_vals < hi)).sum())
                    label = f"[{lo:g}, {hi:g})" if math.isfinite(hi) else f">= {lo:g}"
                    f.write(f"{label:>20} {100.0*count/max(1,current_active_count):13.2f}% {count:14,}\n")

                f.write("\nPercentiles over scored active sequences:\n")
                f.write("-" * 52 + "\n")
                for p in [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]:
                    f.write(f"{'p'+str(p):>20} {float(np.percentile(scored_vals, p)):>20.6f}\n")
                f.write(f"{'mean':>20} {float(scored_vals.mean()):>20.6f}\n")
                f.write(f"{'std':>20} {float(scored_vals.std()):>20.6f}\n")

        # Remove every scored active sequence with perplexity at or below the threshold
        pruning_enabled = cfg.get("pruning_enabled", False)
        prune_threshold = float(cfg.get("pruning_perplexity_threshold", 3.45))

        newly_pruned_mask = np.zeros_like(pruned_mask)
        if pruning_enabled:
            # scored_mask already leaves out pruned sequences and the special
            # values. This matters here: in run_clm.py the values -1 and -2 fail
            # acc >= lower threshold anyway, but here they would pass ppl <= threshold.
            newly_pruned_mask = scored_mask & (ppl_array <= prune_threshold)

        newly_pruned_count = int(newly_pruned_mask.sum())
        pruned_mask = pruned_mask | newly_pruned_mask
        total_pruned = int(pruned_mask.sum())
        active_count = num_train_sequences - total_pruned

        prune_path = os.path.join(tracking_dir, f"epoch_{display_epoch}_pruning_table.txt")
        with open(prune_path, "w") as f:
            f.write(f"Epoch {display_epoch} Pruning Table\n")
            f.write(f"Pruning enabled: {pruning_enabled}\n")
            f.write(f"Rule: remove any active sequence with perplexity <= {prune_threshold:.6f}\n")
            f.write(f"Current active dataset size: {current_active_count}\n")
            f.write(f"Scored active sequences: {scored_idx.size}\n")
            f.write(f"Newly pruned this epoch: {newly_pruned_count}")
            f.write(f"  ({100.0*newly_pruned_count/max(1,current_active_count):.4f}% of active)\n")
            f.write(f"Total pruned so far: {total_pruned}\n")
            f.write(f"Active sequences next epoch: {active_count}\n")

        elapsed_hours = (time.time() - train_start_time) / 3600.0
        logger.info(
            f"CLM pruning, epoch {display_epoch}: elapsed {elapsed_hours:.2f}h, "
            f"GPU-hours {elapsed_hours*world_size:.2f}, newly pruned {newly_pruned_count:,} "
            f"(ppl <= {prune_threshold:.4f}), total {total_pruned:,} "
            f"({100.0*total_pruned/num_train_sequences:.4f}%), active next {active_count:,}")

        np.save(os.path.join(tracking_dir, "pruned_mask.npy"), pruned_mask)
        np.save(os.path.join(tracking_dir, f"epoch_{display_epoch}_ppl.npy"),
                ppl_array.astype(np.float32))
        pruned_tensor = torch.from_numpy(pruned_mask.view(np.uint8)).to(device)
        rank0_ppl_array = ppl_array
    else:
        pruned_tensor = torch.zeros(num_train_sequences, dtype=torch.uint8, device=device)
        rank0_ppl_array = None

    if is_distributed:
        dist.broadcast(pruned_tensor, src=0)
    pruned_mask = pruned_tensor.cpu().numpy().view(bool).copy()
    if is_distributed:
        dist.barrier()

    return pruned_mask, rank0_ppl_array


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
    train_dataset, eval_dataset, num_train_sequences = prepare_datasets(cfg, tokenizer, rank, world_size)
    if rank == 0:
        logger.info(f"Training sequences: {num_train_sequences:,}")
        if eval_dataset is not None:
            logger.info(f"Eval sequences: {len(eval_dataset):,}")

    collator = CLMTrackingDataCollator(pad_token_id=tokenizer.pad_token_id)

    eval_dataloader = None
    if eval_dataset is not None and cfg.get("do_eval", True):
        eval_dataloader = make_eval_dataloader(eval_dataset, collator, cfg, rank, world_size)

    # Model
    model, model_config = build_model(cfg, tokenizer, device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Optimizer: no weight decay on biases, norms, and the Hyena filter
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
    sched_fn = get_cosine_schedule_with_warmup if cfg["lr_scheduler_type"] == "cosine" \
        else get_linear_schedule_with_warmup
    scheduler = sched_fn(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)

    # Pruning / tracking state
    output_dir = cfg["output_dir"]
    tracking_dir = os.path.join(output_dir, cfg["tracking_dir_name"])
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        os.makedirs(tracking_dir, exist_ok=True)

    pruned_mask = np.zeros(num_train_sequences, dtype=bool)
    global_step = 0
    current_epoch = 0
    log_history = []

    ckpt_dir = find_latest_checkpoint(output_dir)
    if ckpt_dir is not None:
        global_step, current_epoch, pruned_mask, log_history = load_checkpoint(
            ckpt_dir, model, optimizer, scheduler, device)
    elif os.path.exists(os.path.join(tracking_dir, "pruned_mask.npy")):
        pruned_mask = np.load(os.path.join(tracking_dir, "pruned_mask.npy"))
        if rank == 0:
            logger.info(f"Loaded existing pruned_mask: {int(pruned_mask.sum()):,} pruned")

    pm_tensor = torch.from_numpy(pruned_mask.view(np.uint8)).to(device)
    dist.broadcast(pm_tensor, src=0)
    pruned_mask = pm_tensor.cpu().numpy().view(bool).copy()

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
        logger.info(f"BF16: {use_bf16} | pruning_enabled: {cfg['pruning_enabled']} "
                    f"| signal: per-sequence perplexity "
                    f"| prune ppl <= {float(cfg['pruning_perplexity_threshold']):.4f}")

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
        epoch_loader, active_count = make_train_dataloader(
            train_dataset, pruned_mask, collator, cfg, rank, world_size, current_epoch)
        steps_this_epoch = len(epoch_loader) // grad_accum
        if rank == 0:
            logger.info(f"Epoch {current_epoch}: {active_count:,} active sequences, "
                        f"{steps_this_epoch:,} steps/epoch (pruned {int(pruned_mask.sum()):,})")

        epoch_buffer = {}        # {line_id: (summed_NLL, summed_predicted_tokens)}
        no_pred_buffer = set()   # line_ids with no predicted positions

        micro_step = 0
        for batch in epoch_loader:
            if global_step >= max_steps:
                break

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            line_ids = batch.pop("line_id")

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])

            loss = outputs.loss
            if grad_accum > 1:
                loss = loss / grad_accum
            loss.backward()

            # per-sequence NLL (the pruning signal; PPL = exp(NLL/tokens))
            # outputs.loss is mean-reduced over the whole batch, so it cannot be
            # reused here. Recompute with reduction="none" and mirror the model's
            # own loss exactly: .float() logits, ignore_index=-100.
            with torch.no_grad():
                logits = outputs.logits
                labels = batch["labels"]
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                valid = shift_labels != -100

                tok_nll = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(shift_labels.shape)
                tok_nll = tok_nll * valid   # redundant with ignore_index; kept for safety

                lids_np = line_ids.cpu().numpy()
                nll_np = tok_nll.sum(dim=-1).double().cpu().numpy()
                total_np = valid.sum(dim=-1).cpu().numpy()

                for i in range(len(lids_np)):
                    lid = int(lids_np[i])
                    n_pred = int(total_np[i])
                    if n_pred == 0:
                        if lid not in epoch_buffer:
                            no_pred_buffer.add(lid)
                        continue
                    seq_nll = float(nll_np[i])
                    if lid in epoch_buffer:
                        prev_s, prev_n = epoch_buffer[lid]
                        epoch_buffer[lid] = (prev_s + seq_nll, prev_n + n_pred)
                    else:
                        epoch_buffer[lid] = (seq_nll, n_pred)
                    no_pred_buffer.discard(lid)

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
                        "active_sequences": int(num_train_sequences - pruned_mask.sum()),
                        "total_pruned": int(pruned_mask.sum()),
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
                    save_checkpoint(model, optimizer, scheduler, global_step, current_epoch,
                                    display_epoch, pruned_mask, epoch_buffer, no_pred_buffer,
                                    num_train_sequences, max_steps, log_history, output_dir,
                                    save_total_limit, tokenizer, rank, train_start_time, world_size)

        # End of epoch: record the perplexities and prune
        if global_step < max_steps:
            pruned_mask, epoch_ppl_array = run_epoch_end_pruning(
                epoch_buffer, no_pred_buffer, pruned_mask, num_train_sequences,
                current_epoch, tracking_dir, device, train_start_time, cfg)

            if rank == 0 and epoch_ppl_array is not None:
                _update_tracked_spreadsheet(tracking_dir, epoch_ppl_array, pruned_mask,
                                            current_epoch, cfg)

        current_epoch += 1

        # Stop after the scheduled number of epochs (counted from 1). The learning-rate
        # schedule is still tied to max_steps; the run just ends earlier.
        graceful_exit_epoch = cfg.get("graceful_exit_epoch", None)
        if graceful_exit_epoch and current_epoch == graceful_exit_epoch:
            if rank == 0:
                exit_dir = os.path.join(output_dir, f"epoch_{graceful_exit_epoch}_checkpoint")
                os.makedirs(exit_dir, exist_ok=True)
                model.module.save_pretrained(exit_dir, safe_serialization=False)
                tokenizer.save_pretrained(exit_dir)
                np.save(os.path.join(exit_dir, "pruned_mask.npy"), pruned_mask)
                logger.info(f"Epoch {graceful_exit_epoch} reached, final model saved to {exit_dir}")
            dist.barrier()
            break

    total_time = (time.time() - train_start_time) / 3600.0
    if rank == 0:
        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE: {total_time:.2f}h wall-clock, {total_time*world_size:.2f} GPU-hours")
        logger.info("=" * 60)
        final_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        model.module.save_pretrained(final_dir, safe_serialization=False)
        tokenizer.save_pretrained(final_dir)
        np.save(os.path.join(final_dir, "pruned_mask.npy"), pruned_mask)
        logger.info(f"Final model saved to: {final_dir}")

    if eval_dataloader is not None:
        metrics = evaluate(model, eval_dataloader, device, use_bf16)
        if rank == 0:
            logger.info(f"FINAL EVAL: {metrics}")

    dist.destroy_process_group()


def _update_tracked_spreadsheet(tracking_dir, epoch_ppl_array, pruned_mask, current_epoch, cfg):
    """Add this epoch's perplexity of 100 tracked sequences to tracked_sequences.xlsx."""
    tracked_xlsx_path = os.path.join(tracking_dir, "tracked_sequences.xlsx")
    tracked_ids_path = os.path.join(tracking_dir, "tracked_sequence_ids.npy")
    try:
        from openpyxl import Workbook, load_workbook
        if not os.path.exists(tracked_ids_path):
            # Pick 100 random sequences from the whole corpus. The choice depends only
            # on the seed, not on the pruning settings, so runs with the same seed
            # follow the same 100 sequences.
            all_ids = np.arange(len(pruned_mask))
            rng = np.random.RandomState(cfg["seed"])
            n_pick = min(100, len(all_ids))
            tracked_ids = np.sort(rng.choice(all_ids, size=n_pick, replace=False))
            np.save(tracked_ids_path, tracked_ids)
            wb = Workbook()
            ws = wb.active
            ws.title = "Tracked Sequences"
            ws.cell(row=1, column=1, value="sequence_id")
            for i, sid in enumerate(tracked_ids):
                ws.cell(row=i + 2, column=1, value=int(sid))
            wb.save(tracked_xlsx_path)

        tracked_ids = np.load(tracked_ids_path)
        wb = load_workbook(tracked_xlsx_path)
        ws = wb.active
        col = ws.max_column + 1
        ws.cell(row=1, column=col, value=f"epoch_{current_epoch + 1}")
        for i, sid in enumerate(tracked_ids):
            row = i + 2
            prev_val = ws.cell(row=row, column=col - 1).value if col > 2 else None
            if prev_val == "PRUNED" or pruned_mask[sid]:
                ws.cell(row=row, column=col, value="PRUNED")
            else:
                ppl = epoch_ppl_array[sid]
                if ppl == SCORE_NOT_TRAINED:
                    ws.cell(row=row, column=col, value="NOT_SEEN")
                elif ppl == SCORE_NO_PRED:
                    ws.cell(row=row, column=col, value="NO_PRED")
                else:
                    ws.cell(row=row, column=col, value=round(float(ppl), 4))
        wb.save(tracked_xlsx_path)
        logger.info(f"Updated tracked_sequences.xlsx for epoch {current_epoch + 1}")
    except ImportError:
        logger.warning("openpyxl not installed, skipping tracked_sequences.xlsx")
    except Exception as e:
        logger.warning(f"Tracked sequences spreadsheet error (non-fatal): {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: torchrun ... run_clm.py <train_config.json>")
        sys.exit(1)
    train(load_config(sys.argv[1]))


if __name__ == "__main__":
    main()
