"""
Fine-tuning of a pretrained Hyena model on one Genomic Benchmarks (GB) task.

It follows dnabert2/finetuning/train.py and writes the same
results/<run_name>/eval_results.json and test_predictions.xlsx. The differences:
  - The model is HyenaDNAForSequenceClassification from ../model_code: the pretrained
    backbone with mean pooling over the sequence and a new linear classification head.
  - The tokenizer is the single-character tokenizer (one token per nucleotide).
  - With --model_max_length -1, the maximum length of a task is the given percentile
    (--max_length_percentile) of its sequence lengths plus 2 special tokens.
  - The model is built at least at the pretraining length, so the Hyena filters are
    used at the same scale as in pretraining.

finetune_gb.sbatch runs this script with the settings used in the paper.

--random_init True builds the model from config.json with random weights instead of
loading the pretrained ones. This is the Full pruned baseline of the paper (no
pretraining at all); the model folder then only supplies the config.
"""

import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch
import transformers
from transformers import TrainerCallback, TrainerState, TrainerControl
import sklearn
import numpy as np
from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

# The HyenaDNA model code and tokenizer live in ../model_code
MODEL_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_code")


def verify_model_loading(model, tokenizer, path, max_len, derived, l_max):
    """Print a short summary of the loaded model and tokenizer."""
    print(f"\n{'='*80}")
    print(f"LOADED FROM: {path}")
    print(f"Model type:      {type(model).__name__}")
    print(f"num_labels:      {model.config.num_labels}")
    print(f"d_model/n_layer: {model.config.d_model} / {model.config.n_layer}")
    print(f"Tokenizer vocab: {tokenizer.vocab_size}")
    print(f"model_max_length: {max_len}  ({'auto P'+'99' if derived else 'fixed'}), "
          f"pretrained l_max: {l_max}")
    if max_len > l_max:
        print(f"NOTE: this task is LONGER than the pretraining context "
              f"({l_max-2} nt). Finetuning at FULL length L={max_len-2} nt; the frozen "
              f"positional embedding is regenerated at this length (HyenaDNA-style "
              f"extrapolation) -- no truncation.")
    try:
        w = model.backbone.embeddings.word_embeddings.weight
        print(f"Embedding weight mean/std: {w.mean().item():.4f} / {w.std().item():.4f} "
              f"(non-trivial => pretrained weights loaded)")
    except Exception:
        print("Could not introspect backbone embedding.")
    print(f"{'='*80}\n")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    random_init: bool = field(default=False, metadata={"help": "Build the model from config.json with random weights instead of loading the pretrained weights (the Full pruned baseline)."})
    hyenadna_code_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Folder with modeling_hyenadna.py, configuration_hyenadna.py and "
                          "tokenization_hyenadna.py. Defaults to ../model_code."})
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="in_proj,out_proj",
                                     metadata={"help": "where to perform LoRA (HyenaDNA has no attention)"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=-1, metadata={"help": "Max length. <=0 => auto P99 from data."})
    max_length_percentile: float = field(default=99.0, metadata={"help": "Percentile for auto max length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps"),
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    use_mcc_selection: bool = field(default=False, metadata={"help": "Use MCC-based selection from top 3 lowest loss checkpoints"})
    top_n_loss_checkpoints: int = field(default=3, metadata={"help": "Number of lowest loss checkpoints to consider for MCC selection"})
    max_steps_cap: int = field(default=0, metadata={"help": "If >0 and num_train_epochs*steps_per_epoch exceeds it, cap training at this many steps (keeps large datasets bounded; 0 disables)."})
    # Default metric for picking the best checkpoint; finetune_gb.sbatch uses the
    # lowest validation loss instead, as in the paper
    metric_for_best_model: str = field(default="matthews_correlation")
    greater_is_better: bool = field(default=True)
    enable_grad_ckpt: bool = field(default=False, metadata={"help": "Turn on gradient checkpointing in the backbone to save memory on long sequences (not needed for the GB tasks of the paper)."})


class MCCBasedCheckpointCallback(TrainerCallback):
    """Optional record of the best checkpoint by MCC among the lowest-loss ones.

    The same as in dnabert2/finetuning/train.py. It only records the choice in
    best_checkpoint_info.json; the evaluated model is still the one the Trainer selects.
    """
    def __init__(self, top_n=3):
        self.top_n = top_n
        self.checkpoint_metrics = []
        self.best_checkpoint = None
        self.best_mcc = -1

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is not None and state.global_step > 0:
            loss = metrics.get('eval_loss', float('inf'))
            mcc = metrics.get('eval_matthews_correlation', -1)
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            self.checkpoint_metrics.append({
                'checkpoint': checkpoint_path, 'step': state.global_step, 'loss': loss, 'mcc': mcc})
            sorted_by_loss = sorted(self.checkpoint_metrics, key=lambda x: x['loss'])[:self.top_n]
            best_among_top = max(sorted_by_loss, key=lambda x: x['mcc'])
            if best_among_top['mcc'] > self.best_mcc:
                self.best_mcc = best_among_top['mcc']
                self.best_checkpoint = best_among_top['checkpoint']
                logging.warning(f"\n{'='*80}")
                logging.warning(f"Top {self.top_n} checkpoints by loss:")
                for i, ckpt in enumerate(sorted_by_loss, 1):
                    marker = " <- BEST MCC" if ckpt['checkpoint'] == self.best_checkpoint else ""
                    logging.warning(f"  {i}. Step {ckpt['step']}: Loss={ckpt['loss']:.4f}, MCC={ckpt['mcc']:.4f}{marker}")
                logging.warning(f"Current best checkpoint: {self.best_checkpoint}")
                logging.warning(f"{'='*80}\n")

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.best_checkpoint:
            best_info_path = os.path.join(args.output_dir, "best_checkpoint_info.json")
            with open(best_info_path, 'w') as f:
                json.dump({'best_checkpoint': self.best_checkpoint, 'best_mcc': self.best_mcc,
                           'all_checkpoints': self.checkpoint_metrics}, f, indent=2)
            logging.warning(f"\nBest checkpoint info saved to: {best_info_path}")
            logging.warning(f"Best checkpoint: {self.best_checkpoint} (MCC: {self.best_mcc:.4f})")


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def derive_max_length(data_path: str, percentile: float, special_tokens: int = 2,
                      cap: Optional[int] = None) -> int:
    """Maximum length of a task: the given percentile of its sequence lengths over
    train, dev and test, plus the special tokens, optionally capped."""
    lengths = []
    for split in ["train.csv", "dev.csv", "test.csv"]:
        p = os.path.join(data_path, split)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            rows = list(csv.reader(f))[1:]
        for r in rows:
            if len(r) >= 3:
                lengths.append(len(r[0]) + len(r[1]))
            elif len(r) >= 2:
                lengths.append(len(r[0]))
    if not lengths:
        return cap or 512
    val = int(np.ceil(np.percentile(lengths, percentile))) + special_tokens
    if cap is not None:
        val = min(val, cap)
    return max(val, 8)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning (the same as in dnabert2/finetuning/train.py)."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, kmer: int = -1):
        super(SupervisedDataset, self).__init__()
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        self.texts = [d[0] if len(data[0]) == 2 else (d[0], d[1]) for d in data]

        # One token per nucleotide, so k-mer input is not supported
        if kmer != -1:
            raise ValueError("HyenaDNA uses single-nucleotide tokenization; set --kmer -1.")

        output = tokenizer(
            texts, return_tensors="pt", padding="longest",
            max_length=tokenizer.model_max_length, truncation=True,
        )
        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.Tensor(labels).long()
        return dict(input_ids=input_ids, labels=labels,
                    attention_mask=input_ids.ne(self.tokenizer.pad_token_id))


def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(valid_labels, valid_predictions),
        "precision": sklearn.metrics.precision_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "recall": sklearn.metrics.recall_score(valid_labels, valid_predictions, average="macro", zero_division=0),
    }


def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return torch.argmax(logits, dim=-1)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)


def save_predictions_to_excel(dataset, predictions, labels, output_path):
    import pandas as pd
    data = {
        'Sequence': [dataset.texts[i] if isinstance(dataset.texts[i], str) else str(dataset.texts[i])
                     for i in range(len(dataset))],
        'True_Label': labels, 'Predicted_Label': predictions}
    pd.DataFrame(data).to_excel(output_path, index=False)
    logging.warning(f"Predictions saved to {output_path}")


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    import sys
    code_dir = model_args.hyenadna_code_dir or MODEL_CODE_DIR
    sys.path.insert(0, code_dir)

    from modeling_hyenadna import HyenaDNAForSequenceClassification
    from configuration_hyenadna import HyenaDNAConfig
    from tokenization_hyenadna import build_dna_tokenizer

    # Config and pretraining length
    config = HyenaDNAConfig.from_pretrained(model_args.model_name_or_path)
    pretrain_l_max = config.l_max

    # Maximum length of this task in tokens (nucleotides + 2 special tokens); it is
    # not capped at the pretraining length
    if training_args.model_max_length is not None and training_args.model_max_length > 0:
        max_len = training_args.model_max_length
        derived = False
    else:
        max_len = derive_max_length(data_args.data_path, training_args.max_length_percentile,
                                    special_tokens=2, cap=None)
        derived = True
    training_args.model_max_length = max_len

    tokenizer = build_dna_tokenizer(model_max_length=max_len)

    # Datasets
    train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                      data_path=os.path.join(data_args.data_path, "train.csv"), kmer=-1)
    val_dataset = SupervisedDataset(tokenizer=tokenizer,
                                    data_path=os.path.join(data_args.data_path, "dev.csv"), kmer=-1)
    test_dataset = SupervisedDataset(tokenizer=tokenizer,
                                     data_path=os.path.join(data_args.data_path, "test.csv"), kmer=-1)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Model: pretrained backbone with a new classification head
    config.num_labels = train_dataset.num_labels
    # The model is built at least at the pretraining length, so the Hyena filters are
    # used at the same scale as in pretraining. It is only made longer for a task with
    # sequences longer than the pretraining windows (none of the GB tasks in the paper).
    config.max_seq_length = max(pretrain_l_max - 2, max_len - 2)
    if model_args.random_init:
        # Full pruned baseline: same architecture, random weights
        model = HyenaDNAForSequenceClassification(config)
    else:
        model = HyenaDNAForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, config=config, ignore_mismatched_sizes=True)

    # Optional gradient checkpointing. It is switched on in the backbone after the
    # model is built, because setting config.gradient_checkpointing before loading
    # makes older transformers versions call a method this model does not have.
    if training_args.enable_grad_ckpt:
        model.backbone.gradient_checkpointing = True

    if model_args.random_init:
        print("Random initialization: the pretrained weights were not loaded (Full pruned baseline)")
    verify_model_loading(model, tokenizer, model_args.model_name_or_path, max_len, derived, pretrain_l_max)

    # Optional cap on the number of training steps (off in the paper: --max_steps_cap 0)
    if training_args.max_steps_cap and training_args.max_steps_cap > 0:
        import math
        world = getattr(training_args, "world_size", 1) or 1
        eff_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world
        steps_per_epoch = max(1, math.ceil(len(train_dataset) / eff_bs))
        full_steps = steps_per_epoch * int(training_args.num_train_epochs)
        if full_steps > training_args.max_steps_cap:
            training_args.max_steps = training_args.max_steps_cap
            logging.warning(f"[max_steps_cap] {int(training_args.num_train_epochs)} epochs = {full_steps} steps "
                            f"(> cap {training_args.max_steps_cap}); capping at {training_args.max_steps_cap} steps "
                            f"(~{training_args.max_steps_cap // steps_per_epoch} epochs over {len(train_dataset)} examples).")
        else:
            logging.warning(f"[max_steps_cap] {int(training_args.num_train_epochs)} epochs = {full_steps} steps "
                            f"(<= cap {training_args.max_steps_cap}); running full epochs.")

    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r, lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout, bias="none",
            task_type="SEQ_CLS", inference_mode=False)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    callbacks = []
    if training_args.use_mcc_selection:
        callbacks.append(MCCBasedCheckpointCallback(top_n=training_args.top_n_loss_checkpoints))
        logging.warning(f"Using MCC-based checkpoint selection (top {training_args.top_n_loss_checkpoints} by loss)")

    trainer = transformers.Trainer(
        model=model, tokenizer=tokenizer, args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=data_collator, callbacks=callbacks)
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if training_args.local_rank not in [-1, 0]:
        logging.warning(f"Process {training_args.local_rank} finished, exiting...")
        return

    logging.warning(f"\nMain process (rank 0) continuing with evaluation...")

    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        os.makedirs(results_path, exist_ok=True)
        best_model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model

        logging.warning(f"\n{'='*80}\nEvaluating on test set...  size={len(test_dataset)}\n{'='*80}\n")
        from torch.utils.data import DataLoader
        test_dataloader = DataLoader(
            test_dataset, batch_size=training_args.per_device_eval_batch_size,
            collate_fn=data_collator, shuffle=False, num_workers=0)

        all_predictions, all_labels = [], []
        best_model.eval()
        logging.warning(f"Running inference on {len(test_dataloader)} batches...")
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                if i % 10 == 0:
                    logging.warning(f"Processing batch {i}/{len(test_dataloader)}...")
                batch = {k: v.to(best_model.device) for k, v in batch.items()}
                outputs = best_model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(batch['labels'].cpu().numpy().tolist())
        logging.warning("Inference complete!")

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        unique, counts = np.unique(all_predictions, return_counts=True)
        logging.warning("\nPrediction distribution:")
        for label, count in zip(unique, counts):
            logging.warning(f"  Class {label}: {count} ({100*count/len(all_predictions):.2f}%)")

        results = calculate_metric_with_sklearn(all_predictions, all_labels)
        results = {f"eval_{k}": v for k, v in results.items()}
        logging.warning(f"\nTest Results:")
        logging.warning(f"  Accuracy: {results.get('eval_accuracy', 0):.4f}")
        logging.warning(f"  MCC:      {results.get('eval_matthews_correlation', 0):.4f}")
        logging.warning(f"  F1:       {results.get('eval_f1', 0):.4f}")

        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        save_predictions_to_excel(test_dataset, all_predictions, all_labels,
                                  os.path.join(results_path, "test_predictions.xlsx"))
        logging.warning(f"\n{'='*80}\nEvaluation complete! Results: {results_path}\n{'='*80}\n")


if __name__ == "__main__":
    train()
