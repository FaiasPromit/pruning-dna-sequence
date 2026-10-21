"""
Fine-tuning of a pretrained NT-500M model on one GUE task (the script used for
GUE in the paper; train_nt18.py is the one used for NT-18).

It follows dnabert2/finetuning/train.py and takes the same arguments, and it
writes the same results/<run_name>/eval_results.json and test_predictions.xlsx.
The differences are that the model is the Hugging Face
EsmForSequenceClassification and that the tokenizer is NTTokenizer, imported
from the model folder. Every sequence is padded to model_max_length + 1 tokens
(model_max_length tokens plus <cls>).

The model folder must contain config.json, the weights (pytorch_model.bin or
model.safetensors), vocab.txt and nt_tokenizer.py, as written by
nucleotide_transformer/pretraining/train_nt.py.

--random_init True builds the model from config.json with random weights instead
of loading the pretrained ones. This is the Full pruned baseline of the paper (no
pretraining at all); the model folder then only supplies the config and tokenizer.
"""

import os
import csv
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Any, Tuple, Union

import torch
import transformers
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers import EsmConfig, EsmForSequenceClassification
import sklearn
import numpy as np
from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    random_init: bool = field(default=False, metadata={"help": "Build the model from config.json with random weights instead of loading the pretrained weights (the Full pruned baseline)."})
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the data folder (contains train.csv, dev.csv, test.csv)."})
    kmer: int = field(default=-1, metadata={"help": "Accepted for the same command line as dnabert2/finetuning/train.py; not used."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum number of tokens, not counting the leading <cls>."})
    fp16: bool = field(default=False)
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
    use_mcc_selection: bool = field(default=False, metadata={"help": "Use MCC-based selection from top N lowest-loss checkpoints."})
    top_n_loss_checkpoints: int = field(default=3)
    remove_checkpoints_after_eval: bool = field(default=True, metadata={"help": "Delete checkpoint-* dirs after eval results are saved."})


def verify_model_loading(model, tokenizer, path):
    print(f"\n{'='*80}")
    print(f"LOADED FROM: {path}")
    print(f"Model type:      {type(model).__name__}")
    print(f"Model vocab:     {model.config.vocab_size}")
    print(f"Tokenizer vocab: {len(tokenizer)}")
    print(f"Max length:      {tokenizer.model_max_length}")
    print(f"{'='*80}\n")


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
            ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            self.checkpoint_metrics.append({
                'checkpoint': ckpt, 'step': state.global_step,
                'loss': loss, 'mcc': mcc,
            })
            sorted_by_loss = sorted(self.checkpoint_metrics, key=lambda x: x['loss'])[:self.top_n]
            best = max(sorted_by_loss, key=lambda x: x['mcc'])
            if best['mcc'] > self.best_mcc:
                self.best_mcc = best['mcc']
                self.best_checkpoint = best['checkpoint']
                logging.warning(f"\n{'='*80}")
                logging.warning(f"Top {self.top_n} checkpoints by loss:")
                for i, c in enumerate(sorted_by_loss, 1):
                    marker = " <- BEST MCC" if c['checkpoint'] == self.best_checkpoint else ""
                    logging.warning(f"  {i}. Step {c['step']}: Loss={c['loss']:.4f}, MCC={c['mcc']:.4f}{marker}")
                logging.warning(f"{'='*80}\n")

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.best_checkpoint:
            with open(os.path.join(args.output_dir, "best_checkpoint_info.json"), 'w') as f:
                json.dump({
                    'best_checkpoint': self.best_checkpoint,
                    'best_mcc': self.best_mcc,
                    'all_checkpoints': self.checkpoint_metrics,
                }, f, indent=2)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


class SupervisedDataset(Dataset):
    """Reads train.csv, dev.csv or test.csv and tokenizes the sequences with NTTokenizer."""

    def __init__(self, data_path: str, tokenizer, kmer: int = -1):
        super().__init__()

        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # NT has no separator token, so the two sequences of a pair are joined.
            # No GUE task used in the paper has sequence pairs.
            logging.warning("Pair classification (concatenating sequences for NT)...")
            texts = [d[0] + d[1] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        # Original sequences, written to the predictions spreadsheet
        self.texts = [d[0] if len(data[0]) == 2 else (d[0], d[1]) for d in data]

        # model_max_length does not count <cls>; encode() adds <cls> and pads or
        # truncates to max_length tokens in total
        max_len = tokenizer.model_max_length + 1
        encoded = tokenizer.batch_encode(texts, max_length=max_len, padding="max_length")

        self.input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: Any  # NTTokenizer, not a Hugging Face PreTrainedTokenizer

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
    p = predictions[valid_mask]; y = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(y, p),
        "f1": sklearn.metrics.f1_score(y, p, average="macro", zero_division=0),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(y, p),
        "precision": sklearn.metrics.precision_score(y, p, average="macro", zero_division=0),
        "recall": sklearn.metrics.recall_score(y, p, average="macro", zero_division=0),
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
        'True_Label': labels,
        'Predicted_Label': predictions,
    }
    pd.DataFrame(data).to_excel(output_path, index=False)
    logging.warning(f"Predictions saved to {output_path}")


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Import NTTokenizer from the model folder
    sys.path.insert(0, model_args.model_name_or_path)
    try:
        from nt_tokenizer import NTTokenizer
    except ImportError as e:
        raise ImportError(
            f"Could not import nt_tokenizer from {model_args.model_name_or_path}. "
            f"The model folder needs nt_tokenizer.py and vocab.txt; copy them from "
            f"nucleotide_transformer/tokenizer/ if they are missing."
        ) from e

    tokenizer = NTTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
    )

    # datasets
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

    # model
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

    # callbacks
    callbacks = []
    if training_args.use_mcc_selection:
        callbacks.append(MCCBasedCheckpointCallback(top_n=training_args.top_n_loss_checkpoints))
        logging.warning(f"Using MCC-based checkpoint selection from top {training_args.top_n_loss_checkpoints} lowest loss checkpoints")

    # trainer
    # tokenizer=None because NTTokenizer is not a Hugging Face tokenizer; the collator
    # above does the padding, so the Trainer does not need it
    trainer = transformers.Trainer(
        model=model,
        tokenizer=None,
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

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if training_args.local_rank not in [-1, 0]:
        logging.warning(f"Process {training_args.local_rank} finished, exiting...")
        return

    # Test-set evaluation (main process only)
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        os.makedirs(results_path, exist_ok=True)

        # With load_best_model_at_end, the Trainer has already loaded the best checkpoint
        # (lowest validation loss, or the metric given by --metric_for_best_model)
        best_model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model

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

        all_predictions, all_labels = [], []
        best_model.eval()
        logging.warning(f"Running inference on {len(test_dataloader)} batches...")
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                if i % 10 == 0:
                    logging.warning(f"Processing batch {i}/{len(test_dataloader)}...")
                batch = {k: v.to(best_model.device) for k, v in batch.items()}
                outputs = best_model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(preds.cpu().numpy().tolist())
                all_labels.extend(batch['labels'].cpu().numpy().tolist())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        unique, counts = np.unique(all_predictions, return_counts=True)
        logging.warning(f"\nPrediction distribution:")
        for label, count in zip(unique, counts):
            logging.warning(f"  Class {label}: {count} samples ({100*count/len(all_predictions):.2f}%)")

        unique_t, counts_t = np.unique(all_labels, return_counts=True)
        logging.warning(f"\nTrue label distribution:")
        for label, count in zip(unique_t, counts_t):
            logging.warning(f"  Class {label}: {count} samples ({100*count/len(all_labels):.2f}%)")

        results = calculate_metric_with_sklearn(all_predictions, all_labels)
        results = {f"eval_{k}": v for k, v in results.items()}

        logging.warning(f"\nTest Results:")
        logging.warning(f"  Accuracy: {results.get('eval_accuracy', 0):.4f}")
        logging.warning(f"  MCC:      {results.get('eval_matthews_correlation', 0):.4f}")
        logging.warning(f"  F1:       {results.get('eval_f1', 0):.4f}")
        logging.warning(f"  Precision:{results.get('eval_precision', 0):.4f}")
        logging.warning(f"  Recall:   {results.get('eval_recall', 0):.4f}")

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
