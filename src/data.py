"""
Data loading utilities for ABM-LoRA

Adapted from LoRA-GA: https://github.com/Outsider565/LoRA-GA
"""

from datasets import load_dataset
import functools
import os
import pickle
import hashlib


def cache_to_disk(root_datadir="data_cache"):
    """Simple disk caching decorator for dataset loading"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not os.path.exists(root_datadir):
                os.makedirs(root_datadir)
            
            # Create cache filename from function name and args
            func_name = func.__name__
            args_str = "_".join(map(str, args))
            kwargs_str = "_".join(f"{k}={v}" for k, v in kwargs.items())
            params_str = f"{args_str}_{kwargs_str}"
            params_hash = hashlib.md5(params_str.encode()).hexdigest()
            
            cache_filename = os.path.join(root_datadir, f"{func_name}_{params_hash}.pkl")
            
            # Load from cache if exists
            if os.path.exists(cache_filename):
                with open(cache_filename, "rb") as f:
                    print(f"Loading cached data for {func_name}")
                    return pickle.load(f)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            with open(cache_filename, "wb") as f:
                pickle.dump(result, f)
                print(f"Cached data for {func_name}")
            
            return result
        
        return wrapper
    return decorator


def load_mrpc():
    """
    Load MRPC dataset from GLUE benchmark.
    Task: Semantic similarity classification (binary)
    """
    dataset = load_dataset("glue", "mrpc")
    instruction = "classify the semantic similarity of the text: "
    label_map = {0: "different", 1: "equivalent", -1: "other"}
    
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence1"]}\n{e["sentence2"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set


@cache_to_disk("data_cache")
def load_mnli():
    """
    Load MNLI dataset from GLUE benchmark.
    Task: Natural language inference (3-way classification)
    """
    dataset = load_dataset("glue", "mnli")
    instruction = "classify the text relationship: "
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction", -1: "other"}
    
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["premise"]}\n{e["hypothesis"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    
    train_set = dataset["train"]
    validation_matched = dataset["validation_matched"]
    return train_set, validation_matched, validation_matched


def load_sst2():
    """
    Load SST-2 dataset from GLUE benchmark.
    Task: Sentiment classification (binary)
    """
    dataset = load_dataset("glue", "sst2")
    instruction = "classify the sentiment of the text: "
    label_map = {0: "negative", 1: "positive", -1: "other"}
    
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set


def load_cola():
    """
    Load CoLA dataset from GLUE benchmark.
    Task: Linguistic acceptability (binary)
    """
    dataset = load_dataset("glue", "cola")
    instruction = "classify the grammatical acceptability of the text: "
    label_map = {0: "unacceptable", 1: "acceptable", -1: "other"}
    
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set


def load_qnli():
    """
    Load QNLI dataset from GLUE benchmark.
    Task: Question-answering NLI (binary)
    """
    dataset = load_dataset("glue", "qnli")
    instruction = "classify whether the question is entailed by the text: "
    label_map = {0: "entailment", 1: "not_entailment", -1: "other"}
    
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["question"]}\n{e["sentence"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set


# Dataset registry
DATASET_MAP = {
    "mrpc": load_mrpc,
    "mnli": load_mnli,
    "sst2": load_sst2,
    "cola": load_cola,
    "qnli": load_qnli,
}
