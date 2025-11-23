"""
Baseline: Standard LoRA Fine-tuning

Train a vanilla LoRA model to use as a teacher for ABM-LoRA Stage 1.
This script provides a baseline implementation without any special initialization.

Usage:
    python -m src.baselines.lora_normal \
        --dataset_name mrpc \
        --lora_rank 8 \
        --lora_alpha 16 \
        --num_epochs 8
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
from fire import Fire
import matplotlib.pyplot as plt
import csv
import wandb
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator

from src.utils import (
    transform_dataset,
    initialize_text_to_text_model,
    find_all_linear_modules,
    train_text_to_text_model,
)
from src.data import DATASET_MAP


def main(
    # Model config
    model_id: str = "t5-base",
    model_dtype: str = "fp32",
    
    # LoRA config
    lora_alpha: int = 16,
    lora_rank: int = 8,
    lora_dropout: float = 0.05,
    
    # Dataset config
    dataset_name: str = "mrpc",
    max_length: int = 128,
    
    # Training config
    num_epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    eval_epochs: int = 1,
    early_stopping_patience: int = 5,
    seed: int = 5,
    
    # Output config
    output_dir: str = "./baseline_normal",
    wandb_project: str = "ABM-LoRA-Baseline",
    wandb_mode: str = "offline",
):
    """
    Train a standard LoRA model as a baseline/teacher.
    
    This creates a vanilla LoRA model that can be used as a teacher
    for ABM-LoRA Stage 1 initialization.
    
    Args:
        model_id: HuggingFace model identifier
        model_dtype: Model dtype ("fp32", "bf16")
        lora_alpha: LoRA scaling parameter
        lora_rank: LoRA rank
        lora_dropout: LoRA dropout rate
        dataset_name: Dataset name from GLUE benchmark
        max_length: Maximum sequence length
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate
        warmup_ratio: Warmup ratio for lr scheduler
        weight_decay: Weight decay
        max_grad_norm: Maximum gradient norm
        eval_epochs: Evaluate every N epochs
        early_stopping_patience: Early stopping patience
        seed: Random seed
        output_dir: Directory to save trained model
        wandb_project: Weights & Biases project name
        wandb_mode: Weights & Biases mode (online/offline)
    """
    
    accelerator = Accelerator()
    
    # Setup wandb
    config = {
        "baseline": "normal_lora",
        "model": model_id,
        "dataset": dataset_name,
        "lora_alpha": lora_alpha,
        "lora_rank": lora_rank,
        "seed": seed,
    }
    wandb_name = "_".join([f"{k}={v}" for k, v in config.items()])
    
    if accelerator.is_local_main_process:
        wandb.init(
            name=wandb_name,
            mode=wandb_mode,
            project=wandb_project,
            config=config,
            reinit=True
        )
    
    print(f"\n{'='*80}")
    print(f"Baseline Training: Standard LoRA")
    print(f"{'='*80}")
    print(f"Model: {model_id}")
    print(f"Dataset: {dataset_name}")
    print(f"LoRA config: rank={lora_rank}, alpha={lora_alpha}")
    print(f"Training: {num_epochs} epochs, lr={learning_rate}")
    print(f"{'='*80}\n")
    
    # Initialize model
    print("📥 Loading model...")
    model, tokenizer = initialize_text_to_text_model(
        model_id,
        "ConditionalGeneration",
        model_dtype,
        flash_attention=False
    )
    
    # Setup LoRA
    print("\n🔧 Setting up LoRA...")
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=find_all_linear_modules(model),
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    
    model = get_peft_model(model, peft_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✅ LoRA model created")
    print(f"   Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Load dataset
    print(f"\n📊 Loading dataset: {dataset_name}")
    dataset_func = DATASET_MAP[dataset_name]
    train_set, val_set, _ = dataset_func()
    
    transform_dataset(
        model_type="ConditionalGeneration",
        tokenizer=tokenizer,
        dataset=train_set,
        max_length=max_length,
    )
    
    print(f"✅ Dataset prepared: {len(train_set)} train, {len(val_set)} val")
    
    # Setup output directory
    save_dir = os.path.join(output_dir, wandb_name)
    os.makedirs(save_dir, exist_ok=True)
    
    if accelerator.is_local_main_process:
        model.save_pretrained(save_dir)
        print(f"💾 Initial model saved to: {save_dir}")
    
    # Loss logging
    loss_log = []
    
    def log_loss_callback(loss, step):
        loss_log.append((step, loss))
        if accelerator.is_local_main_process:
            wandb.log({"train_loss": loss, "step": step})
    
    # Train
    print(f"\n{'='*80}")
    print(f"Starting Training")
    print(f"{'='*80}\n")
    
    model = train_text_to_text_model(
        run_name=os.path.join("baseline_normal", wandb_name),
        train_dataset=train_set,
        valid_dataset=val_set,
        model=model,
        tokenizer=tokenizer,
        model_type="ConditionalGeneration",
        num_train_epochs=num_epochs,
        per_device_batch_size=batch_size,
        real_batch_size=batch_size * accelerator.num_processes,
        bf16=(model_dtype == "bf16"),
        eval_epochs=eval_epochs,
        early_stopping_patience=early_stopping_patience,
        max_length=max_length,
        logging_steps=1,
        learning_rate=learning_rate,
        num_process=accelerator.num_processes,
        gradient_checkpointing=False,
        seed=seed,
        training_args={
            "lr_scheduler_type": "cosine",
            "max_grad_norm": max_grad_norm,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
        },
        log_loss_callback=log_loss_callback,
    )
    
    # Save final model
    if accelerator.is_local_main_process:
        print(f"\n💾 Saving final model...")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        # Save loss history
        if len(loss_log) > 0:
            csv_path = os.path.join(save_dir, "loss_history.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "loss"])
                writer.writerows(loss_log)
            
            # Plot loss curve
            steps, losses = zip(*loss_log)
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses, linewidth=2, label="Training Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title(f"Baseline Training Loss: Standard LoRA ({dataset_name})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
            plt.close()
            
            final_loss = sum([l for _, l in loss_log[-10:]]) / min(10, len(loss_log))
            print(f"✅ Final loss (last 10 steps avg): {final_loss:.6f}")
        
        print(f"\n{'='*80}")
        print(f"✅ Training Complete!")
        print(f"{'='*80}")
        print(f"Model saved to: {save_dir}")
        print(f"Total steps: {len(loss_log)}")
        print(f"\nUse this model as teacher for ABM Stage 1:")
        print(f"  python src/abm/stage1.py \\")
        print(f"      --teacher_lora_path {save_dir} \\")
        print(f"      --dataset_name {dataset_name}")
        print(f"{'='*80}\n")
    
    if accelerator.is_local_main_process:
        wandb.finish()


if __name__ == "__main__":
    Fire(main)
