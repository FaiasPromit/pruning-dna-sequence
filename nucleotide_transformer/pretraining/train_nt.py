#!/usr/bin/env python
"""
NT-500M MLM pretraining on NT-HRG with Sequence Pruning.

The model is the Hugging Face ESM encoder (EsmForMaskedLM) with the NT-500M
configuration, trained from scratch with the 6-mer tokenizer in ../tokenizer.
The recipe follows the NT-500M human reference genome model: 15% masking with
the 80/10/10 rule, Adam (beta1 0.9, beta2 0.999, eps 1e-8), a linear warmup of
the learning rate from start_lr to peak_lr followed by inverse square root
decay with a floor at 1% of the peak, and a random offset of up to 100 bp per
sequence and epoch.

Sequence Pruning works as in dnabert2/pretraining/run_mlm.py: during each epoch
the MLM accuracy of every sequence is counted on its [MASK] positions, and at
the epoch boundary the sequences whose accuracy lies in
[pruning_lower_threshold, pruning_upper_threshold] are removed for the rest of
the run. pruning_window > 1 averages each sequence's accuracy over its last
pruning_window epochs instead. With "pruning_enabled": false nothing is pruned
(the unpruned baseline). Every checkpoint is saved with save_pretrained, so
EsmForMaskedLM.from_pretrained loads it.

Launch with one process per GPU:
    torchrun --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> train_nt.py <config.json>
"""

import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
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
        return token_ids, idx


def collate_fn(batch, pad_id: int = 1, cls_id: int = 3):
    """Prepend <cls> and pad to the longest sequence in the batch. batch is a list of (token_ids, line_id)."""
    max_len = max(len(ids) for ids, _ in batch) + 1  # +1 for CLS
    input_ids = []
    attention_mask = []
    line_ids = []
    for ids, lid in batch:
        seq = [cls_id] + ids
        pad_len = max_len - len(seq)
        attention_mask.append([1] * len(seq) + [0] * pad_len)
        seq += [pad_id] * pad_len
        input_ids.append(seq)
        line_ids.append(lid)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "line_id": torch.tensor(line_ids, dtype=torch.long),
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


def save_checkpoint(
    model, optimizer, scheduler, tokenizer,
    global_step, epoch, cfg, output_dir,
    pruned_mask=None, epoch_buffer=None, no_mask_buffer=None,
    acc_history=None,
):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        model.module.save_pretrained(ckpt_dir, safe_serialization=False)
        tokenizer.save_pretrained(ckpt_dir)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
            },
            os.path.join(ckpt_dir, "training_state.pt"),
        )

        # Pruning state (the partial-epoch buffer is saved for inspection but not reloaded on resume)
        if pruned_mask is not None:
            np.save(os.path.join(ckpt_dir, "pruned_mask.npy"), pruned_mask)
        if acc_history is not None:
            np.save(os.path.join(ckpt_dir, "acc_history.npy"), acc_history)
        if epoch_buffer is not None or no_mask_buffer is not None:
            with open(os.path.join(ckpt_dir, "pruning_buffer.json"), "w") as f:
                json.dump({
                    "epoch_buffer": {str(k): list(v) for k, v in (epoch_buffer or {}).items()},
                    "no_mask_buffer": list(no_mask_buffer or set()),
                }, f)

        logger.info(f"Saved checkpoint: {ckpt_dir}")

        # Rotate old checkpoints
        save_total = cfg.get("save_total_limit", 50)
        ckpts = sorted(
            [
                d
                for d in os.listdir(output_dir)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
            ],
            key=lambda x: int(x.split("-")[1]),
        )
        while len(ckpts) > save_total:
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


def make_train_dataloader(full_dataset, pruned_mask, collate, cfg,
                          rank, world_size, epoch):
    """Build a fresh DataLoader over only the active (unpruned) sequences."""
    active_indices = np.where(~pruned_mask)[0].tolist()
    subset = Subset(full_dataset, active_indices)
    sampler = DistributedSampler(
        subset, num_replicas=world_size, rank=rank,
        shuffle=True, drop_last=True,
        seed=cfg.get("seed", 42) + epoch,
    )
    sampler.set_epoch(epoch)
    loader = DataLoader(
        subset,
        batch_size=cfg["per_device_train_batch_size"],
        sampler=sampler,
        collate_fn=collate,
        num_workers=cfg.get("dataloader_num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    return loader, len(active_indices)


def run_epoch_end_pruning(epoch_buffer, no_mask_buffer, pruned_mask,
                          num_train_sequences, current_epoch,
                          tracking_dir, device, train_start_time, cfg,
                          acc_history):
    """Gather per-sequence accuracy from all ranks and update the pruning mask.

    A sequence is pruned when its accuracy averaged over its last pruning_window
    epochs lies in [pruning_lower_threshold, pruning_upper_threshold]. It can be
    pruned only if each of those epochs gave it an accuracy (it was processed
    and had at least one [MASK] position).

    acc_history: rank-0 float32 array of shape (num_train_sequences, window)
                 holding each sequence's accuracy in its last `window` epochs,
                 with -1.0 for an epoch without one. Ignored on other ranks.

    Returns (updated_pruned_mask, acc_array, avg_array, acc_history); acc_array
    and avg_array are None on ranks other than 0.
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
        display_epoch = current_epoch + 1  # counted from 1 in the tracking files
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

        # Store this epoch's accuracy in the history of the last `window` epochs
        # (-1.0 when the sequence was not processed or had no [MASK] positions)
        window = int(cfg.get("pruning_window", 1))
        slot = current_epoch % window
        valid_this_epoch = acc_array >= 0.0
        acc_history[:, slot] = np.where(valid_this_epoch, acc_array, -1.0)

        # Average over the window. Only sequences with an accuracy in every epoch of
        # the window can be pruned; the others get -1.0 and are kept.
        eligible = (acc_history >= 0.0).all(axis=1)
        avg_array = np.full(num_train_sequences, -1.0, dtype=np.float32)
        avg_array[eligible] = acc_history[eligible].mean(axis=1)

        # Stats on currently-active sequences
        keep_before = ~pruned_mask
        current_line_ids = np.where(keep_before)[0]
        current_active_count = len(current_line_ids)
        current_active_acc = acc_array[current_line_ids]
        current_active_avg = avg_array[current_line_ids]
        tracked_count = len(merged) + int((acc_array == -2.0).sum())

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

        # Prune the sequences whose average accuracy is in the pruning range
        # (sequences with -1.0 are below every range and are kept)
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

        # Write the pruning table, over the same window averages the pruning decision used
        # (-1.0 means the sequence could not be pruned this epoch)
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
    stats = torch.tensor([total_loss, total_correct, total_masked, i + 1], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(stats)

    avg_loss = stats[0].item() / max(stats[3].item(), 1)
    accuracy = stats[1].item() / max(stats[2].item(), 1)
    model.train()
    return {"eval_loss": avg_loss, "eval_accuracy": accuracy}


def main():
    # config
    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = json.load(f)

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

    # model
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    latest_ckpt = find_latest_checkpoint(output_dir) if not cfg.get("overwrite_output_dir", False) else None

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
    if hasattr(model.esm, 'contact_head'):
        del model.esm.contact_head

    model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Model parameters: {n_params:.1f}M")

    # datasets
    train_dataset = NTPretrainDataset(
        cfg["train_file"], tokenizer, max_tokens=cfg.get("max_tokens_per_sequence", 1000)
    )
    eval_dataset = NTPretrainDataset(
        cfg["validation_file"], tokenizer, max_tokens=cfg.get("max_tokens_per_sequence", 1000)
    )

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

    # resume state
    global_step = 0
    start_epoch = 0
    if latest_ckpt:
        state_path = os.path.join(latest_ckpt, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            global_step = state["global_step"]
            start_epoch = state["epoch"]
            if rank == 0:
                logger.info(f"Resumed at step {global_step}, epoch {start_epoch}")

    # pruning state
    num_train_sequences = len(train_dataset)
    tracking_dir = os.path.join(output_dir, "mlm_tracking")
    if rank == 0:
        os.makedirs(tracking_dir, exist_ok=True)

    # Each sequence's accuracy in its last `window` epochs (-1.0 = none); used on rank 0 only
    pruning_window = int(cfg.get("pruning_window", 1))
    acc_history = np.full((num_train_sequences, pruning_window), -1.0, dtype=np.float32)

    pruned_mask = np.zeros(num_train_sequences, dtype=bool)
    if latest_ckpt:
        pm_path = os.path.join(latest_ckpt, "pruned_mask.npy")
        if os.path.exists(pm_path):
            pruned_mask = np.load(pm_path)
            if rank == 0:
                logger.info(f"Loaded pruned_mask: {int(pruned_mask.sum()):,} pruned")
        # Accuracy history; start a fresh one if it is missing or pruning_window changed
        ah_path = os.path.join(latest_ckpt, "acc_history.npy")
        if os.path.exists(ah_path):
            loaded = np.load(ah_path)
            if loaded.shape == (num_train_sequences, pruning_window):
                acc_history = loaded
            elif rank == 0:
                logger.warning(
                    f"acc_history shape {loaded.shape} != "
                    f"{(num_train_sequences, pruning_window)}; reinitializing history"
                )
    elif os.path.exists(os.path.join(tracking_dir, "pruned_mask.npy")):
        # Edge case: tracking dir has a mask but no full checkpoint
        pruned_mask = np.load(os.path.join(tracking_dir, "pruned_mask.npy"))
        if rank == 0:
            logger.info(f"Loaded existing pruned_mask from tracking_dir: {int(pruned_mask.sum()):,} pruned")

    # Send rank 0's pruned_mask to all ranks
    if dist.is_initialized():
        pm_tensor = torch.from_numpy(pruned_mask.view(np.uint8)).to(device)
        dist.broadcast(pm_tensor, src=0)
        pruned_mask = pm_tensor.cpu().numpy().view(bool).copy()

    mask_token_id = tokenizer.mask_token_id

    # Spreadsheet of 100 randomly chosen sequences, picked once from the whole corpus.
    # The choice depends only on the seed, so runs with the same seed follow the same
    # sequences; on resume the saved tracked_sequence_ids.npy is reused.
    if rank == 0:
        tracked_ids_path = os.path.join(tracking_dir, "tracked_sequence_ids.npy")
        tracked_xlsx_path = os.path.join(tracking_dir, "tracked_sequences.xlsx")
        if not os.path.exists(tracked_ids_path):
            try:
                from openpyxl import Workbook
                all_ids = np.arange(num_train_sequences)
                rng = np.random.RandomState(cfg.get("seed", 42))
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
            except ImportError:
                logger.warning("openpyxl not installed, tracked_sequences.xlsx will be skipped")

    if dist.is_initialized():
        dist.barrier()

    # training loop
    max_steps = cfg["max_steps"]
    accum_steps = cfg["gradient_accumulation_steps"]
    batch_size = cfg["per_device_train_batch_size"]
    use_bf16 = cfg.get("bf16", True)
    scaler = None  # bf16 doesn't need GradScaler

    if rank == 0:
        logger.info(f"Effective batch size: {batch_size} x {world_size} x {accum_steps} = "
                     f"{batch_size * world_size * accum_steps} sequences "
                     f"({batch_size * world_size * accum_steps * cfg.get('max_tokens_per_sequence', 1000) / 1e6:.2f}M tokens)")

    model.train()
    epoch = start_epoch
    t0 = time.time()

    while global_step < max_steps:
        train_dataset.set_epoch(epoch)
        dataloader, active_count = make_train_dataloader(
            train_dataset, pruned_mask, _collate, cfg,
            rank, world_size, epoch,
        )
        if rank == 0:
            logger.info(
                f"Epoch {epoch + 1}: {active_count:,} active sequences "
                f"(pruned {int(pruned_mask.sum()):,})"
            )

        # Per-epoch tracking buffers
        epoch_buffer = {}       # {line_id: (correct_count, masked_count)}
        no_mask_buffer = set()  # line_ids that had no [MASK] positions

        optimizer.zero_grad()
        accum_loss = 0.0

        for step_in_epoch, batch in enumerate(dataloader):
            if global_step >= max_steps:
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

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(
                    input_ids=masked_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / accum_steps

            loss.backward()
            accum_loss += loss.item()

            # Per-sequence MLM accuracy, counted on [MASK] positions only
            with torch.no_grad():
                preds = outputs.logits.argmax(dim=-1)
                line_ids = batch["line_id"]
                # Only positions actually replaced by [MASK] (the 80% part) are counted
                mask_pos = (masked_ids == mask_token_id) & (labels != -100)
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

            if (step_in_epoch + 1) % accum_steps == 0:
                # If the gradient norm is not finite (a NaN or Inf in this accumulation
                # cycle), skip the optimizer step instead of writing NaN into the weights.
                # global_step and the learning-rate schedule do not advance for a skipped
                # step. In normal training the norm is always finite and nothing changes.
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.get("max_grad_norm", 1.0)
                )
                if not torch.isfinite(total_norm):
                    if rank == 0:
                        logger.warning(
                            f"Non-finite grad norm ({total_norm.item()}) at "
                            f"global_step={global_step}; skipping this optimizer step"
                        )
                    optimizer.zero_grad()
                    accum_loss = 0.0
                    continue
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if rank == 0 and global_step % cfg.get("logging_steps", 100) == 0:
                    elapsed = time.time() - t0
                    gpu_hours = world_size * elapsed / 3600
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"step={global_step} | loss={accum_loss:.4f} | "
                        f"lr={lr:.2e} | elapsed={elapsed / 3600:.2f}h | "
                        f"gpu-hours={gpu_hours:.2f}"
                    )
                accum_loss = 0.0

                # Eval
                if global_step % cfg.get("eval_steps", 5000) == 0:
                    eval_sampler = DistributedSampler(
                        eval_dataset, num_replicas=world_size, rank=rank, shuffle=False,
                    )
                    eval_loader = DataLoader(
                        eval_dataset,
                        batch_size=cfg.get("per_device_eval_batch_size", 14),
                        sampler=eval_sampler,
                        collate_fn=_collate,
                        num_workers=2,
                        pin_memory=True,
                    )
                    metrics = evaluate(model, eval_loader, cfg, device)
                    if rank == 0:
                        gpu_hours = world_size * (time.time() - t0) / 3600
                        logger.info(
                            f"[Eval step {global_step}] loss={metrics['eval_loss']:.4f} "
                            f"acc={metrics['eval_accuracy']:.4f} | gpu-hours={gpu_hours:.2f}"
                        )

                # Save
                if global_step % cfg.get("save_steps", 5000) == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, tokenizer,
                        global_step, epoch, cfg, output_dir,
                        pruned_mask=pruned_mask,
                        epoch_buffer=epoch_buffer,
                        no_mask_buffer=no_mask_buffer,
                        acc_history=acc_history,
                    )

        # End of epoch: update the pruning mask and the tracked-sequences spreadsheets
        if global_step < max_steps:
            pruned_before_mask = pruned_mask.copy()
            pruned_mask, epoch_acc_array, epoch_avg_array, acc_history = run_epoch_end_pruning(
                epoch_buffer, no_mask_buffer, pruned_mask,
                num_train_sequences, epoch,
                tracking_dir, device, t0, cfg,
                acc_history,
            )

            # Update tracked-sequences spreadsheet (rank 0 only)
            if rank == 0 and epoch_acc_array is not None:
                tracked_xlsx_path = os.path.join(tracking_dir, "tracked_sequences.xlsx")
                tracked_ids_path = os.path.join(tracking_dir, "tracked_sequence_ids.npy")
                if not os.path.exists(tracked_ids_path):
                    # No spreadsheet was created (openpyxl missing), so there is nothing to update
                    pass
                else:
                    try:
                        from openpyxl import Workbook, load_workbook

                        tracked_ids = np.load(tracked_ids_path)
                        wb = load_workbook(tracked_xlsx_path)
                        ws = wb.active

                        # New column for this epoch (counted from 1)
                        col = ws.max_column + 1
                        ws.cell(row=1, column=col, value=f"epoch_{epoch + 1}")

                        for i, sid in enumerate(tracked_ids):
                            row = i + 2
                            if pruned_before_mask[sid]:
                                # Pruned in an earlier epoch, so not trained in this one
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
                        logger.info(f"Updated tracked_sequences.xlsx for epoch {epoch + 1}")

                        # Second spreadsheet: accuracy averaged over the pruning window
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
                        ws_a.cell(row=1, column=col_a, value=f"epoch_{epoch + 1}")
                        for i, sid in enumerate(tracked_ids):
                            row = i + 2
                            if pruned_before_mask[sid]:
                                ws_a.cell(row=row, column=col_a, value="PRUNED")
                            else:
                                avg = epoch_avg_array[sid]
                                if avg < 0.0:
                                    # no accuracy in some epoch of the window, so it cannot be pruned
                                    ws_a.cell(row=row, column=col_a, value="NOT_ENOUGH")
                                else:
                                    ws_a.cell(row=row, column=col_a, value=round(float(avg), 4))
                        wb_a.save(avg_xlsx_path)
                        logger.info(f"Updated tracked_sequences_avg.xlsx for epoch {epoch + 1}")
                    except ImportError:
                        logger.warning("openpyxl not installed, skipping tracked_sequences.xlsx update")
                    except Exception as e:
                        logger.warning(f"Tracked sequences spreadsheet error (non-fatal): {e}")

        epoch += 1
        if rank == 0:
            gpu_hours = world_size * (time.time() - t0) / 3600
            logger.info(f"Epoch {epoch} complete at step {global_step} | gpu-hours={gpu_hours:.2f}")

        # Stop after the scheduled number of epochs
        graceful_exit_epoch = cfg.get("graceful_exit_epoch", None)
        if graceful_exit_epoch and epoch == graceful_exit_epoch:
            if rank == 0:
                exit_dir = os.path.join(output_dir, f"epoch_{graceful_exit_epoch}_checkpoint")
                os.makedirs(exit_dir, exist_ok=True)
                model.module.save_pretrained(exit_dir, safe_serialization=False)
                tokenizer.save_pretrained(exit_dir)
                np.save(os.path.join(exit_dir, "pruned_mask.npy"), pruned_mask)
                logger.info(f"Epoch {graceful_exit_epoch} reached, final model saved to {exit_dir}")
            if dist.is_initialized():
                dist.barrier()
            break

    # Final save
    save_checkpoint(
        model, optimizer, scheduler, tokenizer,
        global_step, epoch, cfg, output_dir,
        pruned_mask=pruned_mask,
        acc_history=acc_history,
    )

    if rank == 0:
        total_time = time.time() - t0
        total_gpu_hours = world_size * total_time / 3600
        logger.info(f"Training finished.")
        logger.info(f"  Total wall time : {total_time / 3600:.2f} h")
        logger.info(f"  Total GPU-hours : {total_gpu_hours:.2f}")
        logger.info(f"  Steps           : {global_step}")
        logger.info(f"  GPUs            : {world_size}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
