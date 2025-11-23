"""
ABM-LoRA Stage 2: Fine-tuning with Initialized Adapters

This script fine-tunes the LoRA adapters initialized by Stage 1 on downstream tasks.

Usage:
    python -m src.abm.stage2 \
        --stage1_model_path ./ab_stage1/[checkpoint] \
        --dataset_name mrpc \
        --num_epochs 8 \
        --learning_rate 3e-4
"""

import torch
import json
import csv
import os
from fire import Fire
import matplotlib.pyplot as plt
from peft import PeftModel, LoraConfig, get_peft_model
from accelerate import Accelerator
import wandb

from ..utils import (
    transform_dataset,
    initialize_text_to_text_model,
    find_all_linear_modules,
    train_text_to_text_model,
)
from ..data import DATASET_MAP


def main(
    # Model config
    model_id: str = "t5-base",
    stage1_model_path: str = None,
    
    # LoRA config
    lora_alpha: int = 16,
    lora_rank: int = 8,
    lora_dropout: float = 0.05,
    
    # Dataset config
    dataset_name: str = "mrpc",
    max_length: int = 128,
    
    # Training config
    learning_rate: float = 3e-4,
    num_epochs: int = 8,
    batch_size: int = 32,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    seed: int = 5,
    
    # Evaluation config
    eval_epochs: int = 1,
    early_stopping_patience: int = 5,
    
    # Output config
    output_dir: str = "./ab_stage2",
    wandb_project: str = "ABM-LoRA-Stage2",
    wandb_mode: str = "offline",
):
    """
    Stage 2: Fine-tune ABM-initialized LoRA adapters on downstream tasks.
    
    Args:
        model_id: HuggingFace model identifier
        stage1_model_path: Path to Stage 1 initialized adapter
        lora_alpha: LoRA scaling parameter
        lora_rank: LoRA rank
        lora_dropout: LoRA dropout rate
        dataset_name: Dataset name from GLUE benchmark
        max_length: Maximum sequence length
        learning_rate: Learning rate for fine-tuning
        num_epochs: Number of training epochs
        batch_size: Training batch size
        warmup_ratio: Warmup ratio for learning rate scheduler
        weight_decay: Weight decay for optimizer
        max_grad_norm: Maximum gradient norm for clipping
        seed: Random seed
        eval_epochs: Evaluate every N epochs
        early_stopping_patience: Early stopping patience
        output_dir: Directory to save fine-tuned model
        wandb_project: Weights & Biases project name
        wandb_mode: Weights & Biases mode (online/offline)
    """
    
    # Validate arguments
    if stage1_model_path is None:
        raise ValueError("stage1_model_path must be provided")
    if not os.path.exists(stage1_model_path):
        raise ValueError(f"stage1_model_path does not exist: {stage1_model_path}")
    
    accelerator = Accelerator()
    
    # Setup wandb
    config = {
        "stage": "stage2_finetuning",
        "model": model_id,
        "dataset": dataset_name,
        "lora_alpha": lora_alpha,
        "lora_rank": lora_rank,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "seed": seed,
    }
    wandb_name = "_".join([f"{k}={v}" for k, v in config.items()])
    
    if accelerator.is_local_main_process:
        wandb.init(
            name=wandb_name,
            mode=wandb_mode,
            project=wandb_project,
            config=config,
            reinit=True,
        )
    
    print(f"{'='*80}")
    print(f"ABM-LoRA Stage 2: Fine-tuning")
    print(f"{'='*80}")
    print(f"Model: {model_id}")
    print(f"Dataset: {dataset_name}")
    print(f"Stage 1 path: {stage1_model_path}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"{'='*80}\n")
    
    # Initialize base model
    print("📥 Loading base model...")
    model, tokenizer = initialize_text_to_text_model(
        model_id,
        "ConditionalGeneration",
        "fp32",
        flash_attention=False
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=find_all_linear_modules(model),
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, peft_config)
    
    # Load Stage 1 adapter
    print(f"\n📥 Loading Stage 1 adapter...")
    try:
        model.load_adapter(stage1_model_path, "default")
        print(f"✅ Successfully loaded ABM-initialized adapter")
    except Exception as e:
        print(f"❌ Failed to load Stage 1 adapter: {str(e)}")
        raise
    
    model.train()
    
    if accelerator.is_local_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Model initialized: {trainable:,} trainable parameters\n")
    
    # Prepare dataset
    print(f"📊 Loading dataset: {dataset_name}")
    dataset_func = DATASET_MAP[dataset_name]
    train_set, val_set, _ = dataset_func()
    
    transform_dataset(
        model_type="ConditionalGeneration",
        tokenizer=tokenizer,
        dataset=train_set,
        max_length=max_length,
    )
    
    print(f"✅ Dataset prepared: {len(train_set)} train, {len(val_set)} val\n")
    
    # Setup save directory
    save_dir = os.path.join(output_dir, wandb_name)
    if accelerator.is_local_main_process:
        print(f"💾 Output directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
    
    # Loss tracking
    loss_log = []
    
    def log_loss_callback(loss, step):
        """Callback to log training loss"""
        loss_log.append((step, loss))
        if accelerator.is_local_main_process:
            wandb.log({"train_loss": loss, "step": step})
    
    # Fine-tune
    print(f"{'='*80}")
    print(f"Starting Fine-tuning")
    print(f"{'='*80}\n")
    
    model = train_text_to_text_model(
        run_name=os.path.join("abm_stage2", wandb_name),
        train_dataset=train_set,
        valid_dataset=val_set,
        model=model,
        tokenizer=tokenizer,
        model_type="ConditionalGeneration",
        num_train_epochs=num_epochs,
        per_device_batch_size=batch_size,
        real_batch_size=batch_size * accelerator.num_processes,
        bf16=False,
        eval_epochs=eval_epochs,
        early_stopping_patience=early_stopping_patience,
        max_length=max_length,
        logging_steps=1,
        use_loraplus=False,
        loraplus_lr_ratio=None,
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
    
    if accelerator.is_local_main_process:
        # Save model and tokenizer
        print(f"\n💾 Saving fine-tuned model...")
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
        
        # Save metadata
        stage_info = {
            "stage": "ABM-LoRA Stage 2",
            "base_model": model_id,
            "stage1_path": stage1_model_path,
            "dataset": dataset_name,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_length": max_length,
        }
        
        with open(os.path.join(save_dir, "training_info.json"), "w") as f:
            json.dump(stage_info, f, indent=2)
        
        # Save loss log
        if loss_log:
            csv_path = os.path.join(save_dir, "loss_history.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "loss"])
                writer.writerows(loss_log)
            
            # Plot loss curve
            steps, losses = zip(*loss_log)
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses, linewidth=2)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("ABM-LoRA Stage 2 Training Loss")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
            plt.close()
            
            print(f"✅ Loss curve saved")
        
        # Verify final model
        print(f"\n🔍 Verifying final model...")
        verification_model, _ = initialize_text_to_text_model(
            model_id,
            "ConditionalGeneration",
            "fp32",
            flash_attention=False
        )
        verification_model = PeftModel.from_pretrained(verification_model, save_dir)
        verification_model.train()
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"✅ Stage 2 Complete!")
        print(f"{'='*80}")
        print(f"Model saved: {save_dir}")
        print(f"Stage 1 source: {stage1_model_path}")
        print(f"Total steps: {len(loss_log)}")
        
        if loss_log:
            initial_loss = loss_log[0][1]
            final_loss = loss_log[-1][1]
            print(f"Loss: {initial_loss:.6f} → {final_loss:.6f} (Δ{final_loss-initial_loss:+.6f})")
        
        print(f"✅ Model ready for evaluation")
        print(f"{'='*80}\n")
        
        wandb.finish()


if __name__ == "__main__":
    Fire(main)
