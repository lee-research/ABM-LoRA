"""
Evaluation script for ABM-LoRA models

Usage:
    python -m src.evaluation \
        --model_path ./ab_stage2/[checkpoint] \
        --dataset_name mrpc \
        --max_samples 100
"""

import torch
import time
import os
from fire import Fire
from peft import PeftModel

from .utils import initialize_text_to_text_model
from .data import DATASET_MAP


def evaluate_model(
    model,
    tokenizer,
    test_dataset,
    max_samples: int = None,
    device: str = "cuda",
):
    """
    Evaluate model on test dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        test_dataset: Test dataset
        max_samples: Maximum number of samples to evaluate (None = all)
        device: Device to use
        
    Returns:
        Dictionary with accuracy and inference time statistics
    """
    model.eval()
    model.to(device)
    
    # Convert to list if needed
    test_data = list(test_dataset) if not isinstance(test_dataset, list) else test_dataset
    
    # Limit samples
    if max_samples and len(test_data) > max_samples:
        test_data = test_data[:max_samples]
    
    correct = 0
    total = 0
    inference_times = []
    
    print(f"\n🔍 Evaluating on {len(test_data)} samples...")
    
    with torch.no_grad():
        for i, sample in enumerate(test_data):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(test_data)}")
            
            # Prepare input
            input_text = sample['x']
            true_label = sample['y'].lower().strip()
            
            # Tokenize
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            # Measure inference time
            start_time = time.time()
            
            # Generate prediction
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                min_new_tokens=1,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            inference_times.append(time.time() - start_time)
            
            # Decode prediction
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            
            # Handle different output formats
            if "result:" in pred_text:
                pred_label = pred_text.split("result:")[-1].strip()
            else:
                pred_label = pred_text
            
            # Map predictions (handle T5's True/False outputs)
            label_mapping = {
                "true": "equivalent",
                "false": "different",
                "1": "equivalent",
                "0": "different",
            }
            final_pred = label_mapping.get(pred_label, pred_label)
            
            # Check correctness
            if final_pred == true_label:
                correct += 1
            
            total += 1
            
            # Debug first few samples
            if i < 3:
                print(f"    Input: {input_text[:50]}...")
                print(f"    Prediction: '{final_pred}' | True: '{true_label}' | {'✓' if final_pred == true_label else '✗'}")
    
    # Compute metrics
    accuracy = 100.0 * correct / total
    avg_time = sum(inference_times) / len(inference_times)
    
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_inference_time": avg_time,
    }
    
    return results


def main(
    model_path: str,
    dataset_name: str = "mrpc",
    model_id: str = "t5-base",
    max_samples: int = None,
    device: str = "cuda",
):
    """
    Evaluate ABM-LoRA model on a dataset.
    
    Args:
        model_path: Path to fine-tuned model checkpoint
        dataset_name: Dataset name (mrpc, mnli, cola, sst2, qnli)
        model_id: Base model identifier
        max_samples: Maximum samples to evaluate (None = all)
        device: Device to use
    """
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    print(f"{'='*80}")
    print(f"ABM-LoRA Evaluation")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Load base model
    print("📥 Loading base model...")
    base_model, tokenizer = initialize_text_to_text_model(
        model_id,
        "ConditionalGeneration",
        "fp32",
        flash_attention=False
    )
    
    # Load LoRA adapter
    print(f"📥 Loading LoRA adapter from {model_path}...")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Load dataset
    print(f"📊 Loading {dataset_name} dataset...")
    dataset_func = DATASET_MAP[dataset_name]
    _, _, test_set = dataset_func()  # train, val, test
    
    # Evaluate
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_set,
        max_samples=max_samples,
        device=device,
    )
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Results on {dataset_name}")
    print(f"{'='*80}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Correct: {results['correct']}/{results['total']}")
    print(f"Avg inference time: {results['avg_inference_time']*1000:.2f}ms")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    Fire(main)
