"""
Utility functions for ABM-LoRA

Adapted from LoRA-GA: https://github.com/Outsider565/LoRA-GA
"""

import torch
import typing as tp
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset
from .logTrainer import LogTrainer


def seed_everything(seed: int):
    """Set random seed for reproducibility"""
    import random
    import os
    import numpy as np
    
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def find_all_linear_modules(model) -> tp.List[str]:
    """
    Find all linear module names suitable for LoRA adaptation.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of module names (e.g., ["q", "v", "wi"])
    """
    linear_cls = torch.nn.Linear
    output_layer_names = ["lm_head", "embed_tokens"]
    
    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls):
            # Skip output layers
            if not any(output_layer in name for output_layer in output_layer_names):
                module_names.add(name.split(".")[-1])
    
    return list(module_names)


def SeqToSeqEncode(example, tokenizer, max_length=None, ignore_masked_token=False):
    """
    Encode examples for sequence-to-sequence models (e.g., T5).
    
    Args:
        example: Dict with "x" (input) and "y" (output) keys
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        ignore_masked_token: Whether to mask padding tokens in labels
        
    Returns:
        Dict with input_ids, attention_mask, labels, decoder_attention_mask
    """
    inputs = tokenizer(
        example["x"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    outputs = tokenizer(
        example["y"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    
    results = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"],
        "decoder_attention_mask": outputs["attention_mask"],
    }
    
    if ignore_masked_token:
        results["labels"][outputs["attention_mask"] == 0] = -100
    
    return results


def preprocess_dataset(
    dataset: tp.Union[Dataset, tp.List[tp.Tuple[str, str]], tp.List[tp.Dict[str, str]]]
) -> Dataset:
    """
    Convert various dataset formats to HuggingFace Dataset.
    
    Args:
        dataset: Input dataset in various formats
        
    Returns:
        HuggingFace Dataset object
    """
    if isinstance(dataset, list) and isinstance(dataset[0], tuple):
        dataset = Dataset.from_pandas(pd.DataFrame(dataset, columns=["x", "y"]))
    elif isinstance(dataset, list) and isinstance(dataset[0], dict):
        dataset = Dataset.from_dict(
            {k: [dic[k] for dic in dataset] for k in dataset[0]}
        )
    elif isinstance(dataset, dict):
        dataset = Dataset.from_dict(dataset)
    elif isinstance(dataset, Dataset):
        pass
    else:
        raise ValueError(f"Unsupported dataset format: {type(dataset)}")
    
    return dataset


def initialize_text_to_text_model(
    model_name: str,
    model_type: str,
    dtype: str,
    tokenizer: str = None,
    flash_attention: bool = False,
):
    """
    Initialize a text-to-text model with tokenizer.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "t5-base")
        model_type: "CausalLM" or "ConditionalGeneration"
        dtype: Model dtype - "fp32", "bf16", "int8", or "nf4"
        tokenizer: Optional custom tokenizer name
        flash_attention: Whether to use flash attention 2
        
    Returns:
        Tuple of (model, tokenizer)
    """
    assert model_type in ["CausalLM", "ConditionalGeneration"], \
        f"model_type must be 'CausalLM' or 'ConditionalGeneration', got {model_type}"
    
    # Select appropriate model class
    auto_model_class = (
        AutoModelForCausalLM if model_type == "CausalLM" 
        else AutoModelForSeq2SeqLM
    )
    
    # Configure model loading
    model_config = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": True,
    }
    
    if flash_attention:
        model_config["attn_implementation"] = "flash_attention_2"
    
    # Set dtype/quantization
    if dtype == "fp32":
        model_config["torch_dtype"] = torch.float32
    elif dtype == "bf16":
        model_config["torch_dtype"] = torch.bfloat16
    elif dtype == "int8":
        model_config["quantization_config"] = {
            "load_in_8bit": True,
            "llm_int8_threshold": 6.0,
        }
    elif dtype == "nf4":
        model_config["quantization_config"] = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        }
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Use fp32, bf16, int8, or nf4")
    
    # Load model and tokenizer
    model = auto_model_class.from_pretrained(**model_config)
    
    if tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Setup special tokens
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        model.resize_token_embeddings(len(tokenizer))
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def transform_dataset(model_type: str, tokenizer, dataset: Dataset, max_length: int):
    """
    Apply tokenization transform to dataset based on model type.
    
    Args:
        model_type: "CausalLM" or "ConditionalGeneration"
        tokenizer: HuggingFace tokenizer
        dataset: HuggingFace Dataset
        max_length: Maximum sequence length
        
    Returns:
        Transformed dataset
    """
    if model_type == "ConditionalGeneration":
        dataset.set_transform(lambda x: SeqToSeqEncode(x, tokenizer, max_length))
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    return dataset


def train_text_to_text_model(
    run_name: str,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    model_type: str,
    num_train_epochs: int = 1,
    per_device_batch_size: int = 1,
    real_batch_size: int = 32,
    max_length: int = None,
    learning_rate: float = 5e-5,
    eval_epochs: int = 1,
    early_stopping_patience: int = 3,
    bf16: bool = False,
    gradient_checkpointing: bool = False,
    seed: int = 42,
    num_process: int = 1,
    logging_steps: int = 10,
    training_args: dict = None,
    log_loss_callback = None,
    **kwargs,
) -> torch.nn.Module:
    """
    Train a text-to-text model using HuggingFace Trainer.
    
    Args:
        run_name: Name for this training run
        train_dataset: Training dataset
        valid_dataset: Validation dataset
        model: Model to train
        tokenizer: Tokenizer
        model_type: "CausalLM" or "ConditionalGeneration"
        num_train_epochs: Number of training epochs
        per_device_batch_size: Batch size per device
        real_batch_size: Effective batch size (with gradient accumulation)
        max_length: Maximum sequence length
        learning_rate: Learning rate
        eval_epochs: Evaluate every N epochs
        early_stopping_patience: Early stopping patience
        bf16: Use bfloat16 training
        gradient_checkpointing: Use gradient checkpointing
        seed: Random seed
        num_process: Number of processes for distributed training
        logging_steps: Log every N steps
        training_args: Additional training arguments
        log_loss_callback: Optional callback for logging loss
        
    Returns:
        Trained model
    """
    # Preprocess datasets
    train_dataset = preprocess_dataset(train_dataset)
    valid_dataset = preprocess_dataset(valid_dataset)
    
    # Calculate gradient accumulation steps
    assert real_batch_size % per_device_batch_size == 0, \
        "real_batch_size must be divisible by per_device_batch_size"
    
    accu_step = real_batch_size // (per_device_batch_size * num_process)
    
    # Transform datasets
    train_dataset = transform_dataset(model_type, tokenizer, train_dataset, max_length)
    valid_dataset = transform_dataset(model_type, tokenizer, valid_dataset, max_length)
    
    # Calculate evaluation steps
    eval_steps = int(len(train_dataset) * eval_epochs) // real_batch_size
    
    # Setup training arguments
    output_dir = f"./results/{run_name}/{seed}"
    
    base_training_args = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_batch_size,
        "per_device_eval_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": accu_step,
        "logging_dir": "./logs",
        "logging_steps": logging_steps,
        "bf16": bf16,
        "gradient_checkpointing": gradient_checkpointing,
        "optim": "adamw_torch",
        "evaluation_strategy": "steps",
        "eval_steps": eval_steps,
        "save_steps": eval_steps,
        "save_strategy": "steps",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "do_eval": True,
        "learning_rate": learning_rate,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "seed": seed,
        "ddp_find_unused_parameters": False,
    }
    
    # Merge with custom training args
    if training_args:
        base_training_args.update(training_args)
    
    training_args_obj = Seq2SeqTrainingArguments(**base_training_args)
    
    # Initialize trainer
    trainer = LogTrainer(
        model=model,
        args=training_args_obj,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=None,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
        ],
        log_loss_callback=log_loss_callback,
    )
    
    # Train
    trainer.train()
    
    return model
