#!/usr/bin/env python
"""
Fine-tuning of a pretrained NT-500M model on one NT-18 task (the script used for
NT-18 in the paper; train_gue.py is the one used for GUE).

It follows dnabert2/finetuning/train.py and takes the same arguments, and it
writes the same results/<run_name>/eval_results.json and test_predictions.xlsx.
The differences are that the model is the Hugging Face
EsmForSequenceClassification (with the unused contact head removed, as in
pretraining) and that the tokenizer is NTTokenizer from
nucleotide_transformer/tokenizer, given a Hugging Face style __call__.
Here model_max_length counts all tokens including <cls>, and sequences are
padded to the longest one in the split.

The model folder must contain config.json, the weights (pytorch_model.bin or
model.safetensors) and vocab.txt, as written by
nucleotide_transformer/pretraining/train_nt.py.

--random_init True builds the model from config.json with random weights instead
of loading the pretrained ones. This is the Full pruned baseline of the paper (no
pretraining at all); the model folder then only supplies the config and vocab.txt.
"""

import os
import sys
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch
import transformers
from transformers import (
    EsmConfig,
    EsmForSequenceClassification,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import sklearn
import numpy as np
from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

# The NT tokenizer lives in ../tokenizer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tokenizer"))
from nt_tokenizer import NTTokenizer


class NTTokenizerHF(NTTokenizer):
    """NTTokenizer with a Hugging Face style __call__, so the dataset and collator
    from dnabert2/finetuning/train.py can be used unchanged.

    Sequence pairs are not supported; every NT-18 task classifies single sequences.
    """

    def __call__(
        self,
        text,
        padding=True,
        truncation=True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Turn the input into a list of strings
        if isinstance(text, str):
            texts = [text]
        elif isinstance(text, list):
            if len(text) > 0 and isinstance(text[0], (list, tuple)):
                raise ValueError(
                    "NTTokenizerHF: sequence-pair input is not supported. "
                    "All NT-18 downstream tasks are single-sequence classification."
                )
            texts = list(text)
        else:
            raise TypeError(f"NTTokenizerHF: unsupported input type {type(text)}")

        if max_length is None:
            max_length = self.model_max_length
        # max_length counts all tokens, including the leading <cls>
        content_max = max(0, max_length - 1)

        encoded: List[List[int]] = []
        for t in texts:
            content = self.tokenize_sequence(t, max_tokens=content_max)
            ids = [self.cls_token_id] + content
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            encoded.append(ids)

        # Padding
        if padding == "max_length":
            target = max_length
        elif padding == "longest" or padding is True:
            target = max(len(ids) for ids in encoded) if encoded else 0
        else:
            target = None  # do_not_pad

        attention_masks: List[List[int]] = []
        if target is not None:
            for i, ids in enumerate(encoded):
                pad_len = target - len(ids)
                attention_masks.append([1] * len(ids) + [0] * pad_len)
                encoded[i] = ids + [self.pad_token_id] * pad_len
        else:
            attention_masks = [[1] * len(ids) for ids in encoded]

        out: Dict[str, Any] = {"input_ids": encoded, "attention_mask": attention_masks}
        if return_tensors == "pt":
            out = {k: torch.tensor(v) for k, v in out.items()}
        return out


def verify_model_loading(model, tokenizer, path):
    """Print a short summary of the loaded model and tokenizer."""
    print(f"\n{'='*80}")
    print(f"LOADED FROM: {path}")
    print(f"Model type: {type(model).__name__}")
    print(f"Model vocab: {model.config.vocab_size}")
    print(f"Tokenizer vocab: {len(tokenizer)}")
    print(f"Max length: {tokenizer.model_max_length}")

    try:
        has_esm = (
            hasattr(model, "esm")
            and hasattr(model.esm, "encoder")
            and hasattr(model.esm.encoder.layer[0].attention.self, "query")
        )
        print(f"ESM architecture: {'yes' if has_esm else 'no'}")
    except Exception:
        print(f"ESM architecture: could not check")

    # The contact head is removed after loading, as in pretraining
    has_contact = hasattr(model, "esm") and hasattr(model.esm, "contact_head")
    print(f"Contact head removed: {'yes' if not has_contact else 'no'}")

    print(f"{'='*80}\n")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    random_init: bool = field(default=False, metadata={"help": "Build the model from config.json with random weights instead of loading the pretrained weights (the Full pruned baseline)."})
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
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
    remove_checkpoints_after_eval: bool = field(default=True, metadata={"help": "Delete checkpoint-* dirs after eval results are saved."})


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
            loss = metrics.get("eval_loss", float("inf"))
            mcc = metrics.get("eval_matthews_correlation", -1)

            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            self.checkpoint_metrics.append(
                {"checkpoint": checkpoint_path, "step": state.global_step, "loss": loss, "mcc": mcc}
            )

            sorted_by_loss = sorted(self.checkpoint_metrics, key=lambda x: x["loss"])[: self.top_n]
            best_among_top = max(sorted_by_loss, key=lambda x: x["mcc"])

            if best_among_top["mcc"] > self.best_mcc:
                self.best_mcc = best_among_top["mcc"]
                self.best_checkpoint = best_among_top["checkpoint"]

                logging.warning(f"\n{'='*80}")
                logging.warning(f"Top {self.top_n} checkpoints by loss:")
                for i, ckpt in enumerate(sorted_by_loss, 1):
                    marker = " <- BEST MCC" if ckpt["checkpoint"] == self.best_checkpoint else ""
                    logging.warning(
                        f"  {i}. Step {ckpt['step']}: Loss={ckpt['loss']:.4f}, MCC={ckpt['mcc']:.4f}{marker}"
                    )
                logging.warning(f"Current best checkpoint: {self.best_checkpoint}")
                logging.warning(f"{'='*80}\n")

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.best_checkpoint:
            best_info_path = os.path.join(args.output_dir, "best_checkpoint_info.json")
            with open(best_info_path, "w") as f:
                json.dump(
                    {
                        "best_checkpoint": self.best_checkpoint,
                        "best_mcc": self.best_mcc,
                        "all_checkpoints": self.checkpoint_metrics,
                    },
                    f,
                    indent=2,
                )
            logging.warning(f"\nBest checkpoint info saved to: {best_info_path}")
            logging.warning(f"Best checkpoint: {self.best_checkpoint} (MCC: {self.best_mcc:.4f})")


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([MAP[c] for c in sequence])


def generate_kmer_str(sequence: str, k: int) -> str:
    return " ".join([sequence[i : i + k] for i in range(len(sequence) - k + 1)])


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
    return kmer


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer, kmer: int = -1):
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

        if kmer != -1:
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
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
    """Collate examples for supervised fine-tuning."""

    tokenizer: Any  # NTTokenizerHF

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
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
    """Save test predictions to Excel file."""
    import pandas as pd

    data = {
        "Sequence": [
            dataset.texts[i] if isinstance(dataset.texts[i], str) else str(dataset.texts[i])
            for i in range(len(dataset))
        ],
        "True_Label": labels,
        "Predicted_Label": predictions,
    }
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)
    logging.warning(f"Predictions saved to {output_path}")


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Tokenizer (NTTokenizerHF, not AutoTokenizer)
    vocab_file = os.path.join(model_args.model_name_or_path, "vocab.txt")
    if not os.path.isfile(vocab_file):
        raise FileNotFoundError(
            f"vocab.txt not found in {model_args.model_name_or_path}; "
            f"the model folder should be one saved by train_nt.py."
        )
    tokenizer = NTTokenizerHF(
        vocab_file=vocab_file,
        model_max_length=training_args.model_max_length,
    )

    # Datasets
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "train.csv"),
        kmer=data_args.kmer,
    )
    val_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "dev.csv"),
        kmer=data_args.kmer,
    )
    test_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "test.csv"),
        kmer=data_args.kmer,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Model (Hugging Face EsmForSequenceClassification)
    config = EsmConfig.from_pretrained(model_args.model_name_or_path)
    config.num_labels = train_dataset.num_labels

    if model_args.random_init:
        # Full pruned baseline: same architecture, random weights
        model = EsmForSequenceClassification(config)
    else:
        model = EsmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )

    # Remove the unused contact head, as in pretraining
    if hasattr(model, "esm") and hasattr(model.esm, "contact_head"):
        del model.esm.contact_head

    if model_args.random_init:
        print("Random initialization: the pretrained weights were not loaded (Full pruned baseline)")
    verify_model_loading(model, tokenizer, model_args.model_name_or_path)

    # LoRA (off by default and not used in the paper)
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Callbacks
    callbacks = []
    if training_args.use_mcc_selection:
        mcc_callback = MCCBasedCheckpointCallback(top_n=training_args.top_n_loss_checkpoints)
        callbacks.append(mcc_callback)
        logging.warning(
            f"Using MCC-based checkpoint selection from top "
            f"{training_args.top_n_loss_checkpoints} lowest loss checkpoints"
        )

    # Trainer
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # Synchronize all processes after training
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Only the main process evaluates on the test set
    if training_args.local_rank not in [-1, 0]:
        logging.warning(f"Process {training_args.local_rank} finished, exiting...")
        return

    logging.warning(f"\nMain process (rank 0) continuing with evaluation...")

    # Test-set evaluation
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        os.makedirs(results_path, exist_ok=True)

        best_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model

        logging.warning(f"\n{'='*80}")
        logging.warning(f"Evaluating on test set...")
        logging.warning(f"Test dataset size: {len(test_dataset)}")
        logging.warning(f"{'='*80}\n")

        from torch.utils.data import DataLoader

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            shuffle=False,
            num_workers=0,
        )

        all_predictions = []
        all_labels = []

        best_model.eval()
        logging.warning(f"Running inference on {len(test_dataloader)} batches...")

        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                if i % 10 == 0:
                    logging.warning(f"Processing batch {i}/{len(test_dataloader)}...")
                batch = {k: v.to(best_model.device) for k, v in batch.items()}
                outputs = best_model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(batch["labels"].cpu().numpy().tolist())

        logging.warning(f"Inference complete!")

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        logging.warning(f"\nPredicted classes shape: {all_predictions.shape}")
        logging.warning(f"Sample predicted classes (first 20): {all_predictions[:20]}")

        unique, counts = np.unique(all_predictions, return_counts=True)
        logging.warning(f"\nPrediction distribution:")
        for label, count in zip(unique, counts):
            logging.warning(f"  Class {label}: {count} samples ({100*count/len(all_predictions):.2f}%)")

        unique_true, counts_true = np.unique(all_labels, return_counts=True)
        logging.warning(f"\nTrue label distribution:")
        for label, count in zip(unique_true, counts_true):
            logging.warning(f"  Class {label}: {count} samples ({100*count/len(all_labels):.2f}%)")

        results = calculate_metric_with_sklearn(all_predictions, all_labels)
        results = {f"eval_{k}": v for k, v in results.items()}

        logging.warning(f"\nTest Results:")
        logging.warning(f"  Accuracy: {results.get('eval_accuracy', 0):.4f}")
        logging.warning(f"  MCC: {results.get('eval_matthews_correlation', 0):.4f}")
        logging.warning(f"  F1: {results.get('eval_f1', 0):.4f}")
        logging.warning(f"  Precision: {results.get('eval_precision', 0):.4f}")
        logging.warning(f"  Recall: {results.get('eval_recall', 0):.4f}")

        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        excel_path = os.path.join(results_path, "test_predictions.xlsx")
        save_predictions_to_excel(test_dataset, all_predictions, all_labels, excel_path)

        # Delete the checkpoint-* folders; the best weights are already in memory
        if getattr(training_args, "remove_checkpoints_after_eval", True):
            import shutil, glob
            for ckpt in glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")):
                if os.path.isdir(ckpt):
                    shutil.rmtree(ckpt, ignore_errors=True)
                    logging.warning(f"Removed checkpoint dir: {ckpt}")

        logging.warning(f"\n{'='*80}")
        logging.warning(f"Evaluation complete! Results saved to {results_path}")
        logging.warning(f"Excel file: {excel_path}")
        logging.warning(f"{'='*80}\n")


if __name__ == "__main__":
    train()
