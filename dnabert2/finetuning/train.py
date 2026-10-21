"""
Fine-tuning of a pretrained DNABERT-2 model on one GUE or NT-18 task.

Adapted from the DNABERT-2 fine-tuning script
(https://github.com/MAGICS-LAB/DNABERT_2/blob/main/finetune/train.py). The main changes:
  - The model code (bert_layers.py, configuration_bert.py) is imported from the model
    folder itself, and everything is loaded from local files only.
  - After training, the best checkpoint is evaluated on test.csv; the metrics go to
    results/<run_name>/eval_results.json and the predictions to test_predictions.xlsx.
  - The checkpoint-* folders are deleted after evaluation (remove_checkpoints_after_eval).
  - --random_init True builds the model from config.json with random weights instead of
    loading the pretrained ones. This is the Full pruned baseline of the paper (no
    pretraining at all); the model folder then only supplies the config, the tokenizer
    and the model code.

The data folder must contain train.csv, dev.csv and test.csv with a header row and
the columns sequence,label. finetune_gue.sbatch and finetune_nt18.sbatch run this
script with the settings used in the paper.
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

def verify_model_loading(model, tokenizer, path):
    """Print a short summary of the loaded model and tokenizer."""
    print(f"\n{'='*80}")
    print(f"LOADED FROM: {path}")
    print(f"Model type: {type(model).__name__}")
    print(f"Model vocab: {model.config.vocab_size}")
    print(f"Tokenizer vocab: {len(tokenizer)}")
    print(f"Max length: {tokenizer.model_max_length}")
    
    # The DNABERT-2 attention layer has a fused Wqkv projection; a standard BERT does not
    try:
        has_custom = hasattr(model.bert.encoder.layer[0].attention.self, 'Wqkv')
        print(f"DNABERT-2 attention layers: {'yes' if has_custom else 'no'}")
    except:
        print(f"DNABERT-2 attention layers: could not check")
    
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
    use_mcc_selection: bool = field(default=False, metadata={"help": "Log which of the lowest-loss checkpoints has the highest MCC (see MCCBasedCheckpointCallback)"})
    top_n_loss_checkpoints: int = field(default=3, metadata={"help": "Number of lowest-loss checkpoints to compare by MCC"})
    remove_checkpoints_after_eval: bool = field(default=True, metadata={"help": "Delete checkpoint-* dirs after eval results are saved."})
    

class MCCBasedCheckpointCallback(TrainerCallback):
    """Optional record of the best checkpoint by MCC among the lowest-loss ones.

    After each evaluation, it takes the top_n checkpoints with the lowest validation
    loss and notes the one with the highest validation MCC; at the end of training it
    writes this to best_checkpoint_info.json. It only records the choice: the model
    evaluated on the test set is still the one the Trainer selects
    (load_best_model_at_end with metric_for_best_model).
    """
    def __init__(self, top_n=3):
        self.top_n = top_n
        self.checkpoint_metrics = []  # one entry per evaluation: checkpoint, step, loss, mcc
        self.best_checkpoint = None
        self.best_mcc = -1
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is not None and state.global_step > 0:
            loss = metrics.get('eval_loss', float('inf'))
            mcc = metrics.get('eval_matthews_correlation', -1)
            
            # Store checkpoint info
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            self.checkpoint_metrics.append({
                'checkpoint': checkpoint_path,
                'step': state.global_step,
                'loss': loss,
                'mcc': mcc
            })
            
            # Sort by loss (ascending) and get top N
            sorted_by_loss = sorted(self.checkpoint_metrics, key=lambda x: x['loss'])[:self.top_n]
            
            # Among top N lowest loss, find highest MCC
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
            # Save info about best checkpoint
            best_info_path = os.path.join(args.output_dir, "best_checkpoint_info.json")
            with open(best_info_path, 'w') as f:
                json.dump({
                    'best_checkpoint': self.best_checkpoint,
                    'best_mcc': self.best_mcc,
                    'all_checkpoints': self.checkpoint_metrics
                }, f, indent=2)
            logging.warning(f"\nBest checkpoint info saved to: {best_info_path}")
            logging.warning(f"Best checkpoint: {self.best_checkpoint} (MCC: {self.best_mcc:.4f})")


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def get_alter_of_dna_sequence(sequence: str):
    """Complement of a DNA sequence (not reversed)."""
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([MAP[c] for c in sequence])


def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate the k-mer string of each DNA sequence.

    Generated k-mer strings are saved next to the data file, with the suffix _{k}mer.json.
    DNABERT-2 uses BPE tokens, so the sbatch files pass --kmer -1 and this is not used.
    """
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

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")
        
        # Store original texts for later use
        self.texts = [d[0] if len(data[0]) == 2 else (d[0], d[1]) for d in data]
        
        if kmer != -1:
            # only write file on the first process
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

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
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
    """Accuracy, macro F1, MCC, macro precision and macro recall with sklearn."""
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
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

# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)


def compute_metrics(eval_pred):
    """Metrics reported by the Hugging Face Trainer at each evaluation."""
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)


def save_predictions_to_excel(dataset, predictions, labels, output_path):
    """Save test predictions to Excel file."""
    import pandas as pd
    
    # Create dataframe with sequences and predictions
    data = {
        'Sequence': [dataset.texts[i] if isinstance(dataset.texts[i], str) else str(dataset.texts[i]) for i in range(len(dataset))],
        'True_Label': labels,
        'Predicted_Label': predictions
    }
    
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)
    logging.warning(f"Predictions saved to {output_path}")


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Import the model code (bert_layers.py, configuration_bert.py) from the model folder.
    # The pretraining scripts copy it into every model folder they save.
    import sys
    sys.path.insert(0, model_args.model_name_or_path)
    from bert_layers import BertForSequenceClassification
    from configuration_bert import BertConfig
    
    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=False,
        local_files_only=True,
    )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=os.path.join(data_args.data_path, "train.csv"), 
                                      kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "dev.csv"), 
                                     kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "test.csv"), 
                                     kmer=data_args.kmer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Load config and weights from the model folder
    config = BertConfig.from_pretrained(model_args.model_name_or_path)
    config.num_labels = train_dataset.num_labels
    
    if model_args.random_init:
        # Full pruned baseline: same architecture, random weights
        model = BertForSequenceClassification(config)
    else:
        model = BertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )

    if model_args.random_init:
        print("Random initialization: the pretrained weights were not loaded (Full pruned baseline)")
    verify_model_loading(model, tokenizer, model_args.model_name_or_path)

    # configure LoRA (not used in the paper, which fine-tunes all parameters)
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

    # Setup callbacks
    callbacks = []
    if training_args.use_mcc_selection:
        mcc_callback = MCCBasedCheckpointCallback(top_n=training_args.top_n_loss_checkpoints)
        callbacks.append(mcc_callback)
        logging.warning(f"Using MCC-based checkpoint selection from top {training_args.top_n_loss_checkpoints} lowest loss checkpoints")

    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator,
                                   callbacks=callbacks)
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
    
    # Evaluate on the test set and save the results
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        os.makedirs(results_path, exist_ok=True)
        
        # With load_best_model_at_end, the Trainer has already loaded the best checkpoint
        # (lowest validation loss, or the metric given by --metric_for_best_model)
        best_model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model 
        
        # Manual evaluation
        logging.warning(f"\n{'='*80}")
        logging.warning(f"Evaluating on test set...")
        logging.warning(f"Test dataset size: {len(test_dataset)}")
        logging.warning(f"{'='*80}\n")
        
        # Create dataloader
        from torch.utils.data import DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            shuffle=False,
            num_workers=0,  # Avoid multiprocessing issues
        )
        
        # Run inference
        all_predictions = []
        all_labels = []
        
        best_model.eval()
        logging.warning(f"Running inference on {len(test_dataloader)} batches...")
        
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                if i % 10 == 0:
                    logging.warning(f"Processing batch {i}/{len(test_dataloader)}...")
                
                # Move batch to device
                batch = {k: v.to(best_model.device) for k, v in batch.items()}
                
                # Get predictions
                outputs = best_model(**batch)
                logits = outputs.logits
                
                # Store predictions and labels
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(batch['labels'].cpu().numpy().tolist())
        
        logging.warning(f"Inference complete!")
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        logging.warning(f"\nPredicted classes shape: {all_predictions.shape}")
        logging.warning(f"Sample predicted classes (first 20): {all_predictions[:20]}")
        
        # Log prediction distribution
        unique, counts = np.unique(all_predictions, return_counts=True)
        logging.warning(f"\nPrediction distribution:")
        for label, count in zip(unique, counts):
            logging.warning(f"  Class {label}: {count} samples ({100*count/len(all_predictions):.2f}%)")
        
        # Log true label distribution
        unique_true, counts_true = np.unique(all_labels, return_counts=True)
        logging.warning(f"\nTrue label distribution:")
        for label, count in zip(unique_true, counts_true):
            logging.warning(f"  Class {label}: {count} samples ({100*count/len(all_labels):.2f}%)")
        
        # Calculate metrics
        results = calculate_metric_with_sklearn(all_predictions, all_labels)
        results = {f"eval_{k}": v for k, v in results.items()}
        
        logging.warning(f"\nTest Results:")
        logging.warning(f"  Accuracy: {results.get('eval_accuracy', 0):.4f}")
        logging.warning(f"  MCC: {results.get('eval_matthews_correlation', 0):.4f}")
        logging.warning(f"  F1: {results.get('eval_f1', 0):.4f}")
        logging.warning(f"  Precision: {results.get('eval_precision', 0):.4f}")
        logging.warning(f"  Recall: {results.get('eval_recall', 0):.4f}")
        
        # Save results
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Save to Excel
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