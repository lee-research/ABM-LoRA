"""
ABM-LoRA Stage 1: Activation Boundary Matching Initialization

This script performs activation boundary matching to initialize LoRA adapters
by aligning their activation patterns with a pretrained teacher model.

Usage:
    # Run as a module (Recommended)
    python -m src.abm.stage1 \
        --teacher_lora_path ./pretrained_lora/t5-base_mrpc \
        --dataset_name mrpc \
        --lora_rank 8 \
        --lora_alpha 16 \
        --sample_size 128 \
        --max_steps 100
"""

import torch
from fire import Fire
import os
import csv
import time
import matplotlib.pyplot as plt
import wandb
from peft import PeftModel, LoraConfig, get_peft_model
from accelerate import Accelerator

from ..utils import (
    transform_dataset,
    initialize_text_to_text_model,
    find_all_linear_modules,
)
from ..data import DATASET_MAP
from .utils import (
    register_hooks, 
    remove_hooks, 
    compute_matching_accuracy,
    compute_ab_loss,
    teacher_acts,
    student_acts,
    get_t5_layer_config,
    clear_activations,
    get_activation_stats,
    validate_model_paths,
    setup_key_mapping
)


def main(
    # Model config
    model_id: str = "t5-base",
    teacher_lora_path: str = None,
    
    # LoRA config
    lora_alpha: int = 16,
    lora_rank: int = 8,
    lora_dropout: float = 0.05,
    
    # Dataset config
    dataset_name: str = "mrpc",
    sample_size: int = 128,
    max_length: int = 512,
    
    # Training config
    learning_rate: float = 3e-4,
    num_epochs: int = 1,
    max_steps: int = 100,
    seed: int = 5,
    
    # ABM config
    num_layers: int = 6,
    target_type: str = "wi",
    use_decoder: bool = True,
    margin: float = 0.5,
    
    # Output config
    output_dir: str = "./ab_stage1",
    wandb_project: str = "ABM-LoRA-Stage1",
    wandb_mode: str = "offline",
):
    """
    Stage 1: Activation Boundary Matching initialization for LoRA adapters.
    
    Args:
        model_id: HuggingFace model identifier (default: "t5-base")
        teacher_lora_path: Path to pretrained LoRA adapter (teacher model)
        lora_alpha: LoRA scaling parameter
        lora_rank: LoRA rank
        lora_dropout: LoRA dropout rate
        dataset_name: Dataset name from GLUE benchmark
        sample_size: Number of samples to use for initialization
        max_length: Maximum sequence length
        learning_rate: Learning rate for ABM optimization
        num_epochs: Number of epochs
        max_steps: Maximum training steps (early stopping)
        seed: Random seed
        num_layers: Number of layers to apply ABM
        target_type: Target layer type ("wi" for DenseReluDense)
        use_decoder: Whether to include decoder layers
        margin: Margin for hinge loss
        output_dir: Directory to save initialized adapter
        wandb_project: Weights & Biases project name
        wandb_mode: Weights & Biases mode (online/offline)
    """
    
    # Validate arguments
    if teacher_lora_path is None:
        raise ValueError("teacher_lora_path must be provided")
    if not os.path.exists(teacher_lora_path):
        raise ValueError(f"teacher_lora_path does not exist: {teacher_lora_path}")
    
    accelerator = Accelerator()
    
    # Setup wandb
    config = {
        "stage": "stage1_abm_init",
        "model": model_id,
        "dataset": dataset_name,
        "lora_alpha": lora_alpha,
        "lora_rank": lora_rank,
        "sample_size": sample_size,
        "seed": seed,
        "num_layers": num_layers,
        "margin": margin,
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
    
    print(f"{'='*80}")
    print(f"ABM-LoRA Stage 1: Activation Boundary Matching Initialization")
    print(f"{'='*80}")
    print(f"Model: {model_id}")
    print(f"Dataset: {dataset_name}")
    print(f"Teacher LoRA: {teacher_lora_path}")
    print(f"LoRA config: rank={lora_rank}, alpha={lora_alpha}")
    print(f"ABM config: layers={num_layers}, margin={margin}")
    print(f"{'='*80}\n")
    
    # Initialize teacher model (frozen)
    print("📥 Loading teacher model...")
    teacher, _ = initialize_text_to_text_model(
        model_id, 
        "ConditionalGeneration", 
        "fp32", 
        flash_attention=False
    )
    teacher = PeftModel.from_pretrained(teacher, teacher_lora_path)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"✅ Teacher model loaded: {type(teacher).__name__}")
    
    # Initialize student model with LoRA
    print("\n📥 Initializing student model with LoRA...")
    student, tokenizer = initialize_text_to_text_model(
        model_id,
        "ConditionalGeneration",
        "fp32",
        flash_attention=False
    )
    
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=find_all_linear_modules(student),
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    student = get_peft_model(student, peft_config)
    print(f"✅ Student model initialized: {type(student).__name__}")
    print(f"   Trainable params: {student.num_parameters(only_trainable=True):,}")
    
    # Prepare dataset
    print(f"\n📊 Loading dataset: {dataset_name}")
    dataset_func = DATASET_MAP[dataset_name]
    train_set, val_set, _ = dataset_func()
    transform_dataset(
        model_type="ConditionalGeneration",
        tokenizer=tokenizer,
        dataset=train_set,
        max_length=max_length
    )
    
    # Sample subset
    if len(train_set) > sample_size:
        torch.manual_seed(seed)
        indices = torch.randperm(len(train_set))[:sample_size]
        train_set = torch.utils.data.Subset(train_set, indices)
    print(f"✅ Dataset prepared: {len(train_set)} samples")
    
    dataloader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=1, 
        shuffle=True
    )
    
    # Configure layers for ABM
    print(f"\n🔧 Configuring ABM layers...")
    teacher_layers, student_layers, layer_weights = get_t5_layer_config(
        num_layers=num_layers,
        target_type=target_type,
        use_decoder=use_decoder,
        teacher_model=teacher,
        student_model=student
    )
    print(f"✅ Target layers: {len(teacher_layers)} layers")
    print(f"   Type: {target_type}, Decoder: {use_decoder}")
    
    # Validate paths
    valid_teacher_paths, _ = validate_model_paths(teacher, teacher_layers, "Teacher")
    valid_student_paths, _ = validate_model_paths(student, student_layers, "Student")
    
    if len(valid_teacher_paths) == 0 or len(valid_student_paths) == 0:
        raise RuntimeError("No valid teacher/student paths found!")
    
    teacher_layers = valid_teacher_paths
    student_layers = valid_student_paths
    layer_weights = layer_weights[:len(teacher_layers)]
    
    setup_key_mapping(teacher_layers, student_layers)
    
    # Register hooks
    print(f"\n🔗 Registering forward hooks...")
    teacher_hooks = register_hooks(teacher, teacher_layers, is_teacher=True)
    student_hooks = register_hooks(student, student_layers, is_teacher=False)
    print(f"✅ Hooks registered: Teacher={len(teacher_hooks)}, Student={len(student_hooks)}")
    
    # Test hooks
    print("\n🧪 Testing hook functionality...")
    try:
        test_batch = next(iter(dataloader))
        clear_activations()
        
        with torch.no_grad():
            _ = teacher(**test_batch)
        _ = student(**test_batch)
        
        test_stats = get_activation_stats()
        print(f"✅ Hook test passed: {test_stats['common_keys']} common keys")
        
        if test_stats['common_keys'] == 0:
            print("⚠️  Warning: No common keys detected!")
        
        clear_activations()
        
    except Exception as e:
        print(f"❌ Hook test failed: {str(e)}")
        raise
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    
    # Track initial LoRA state
    first_lora_module = None
    lora_B_init = None
    for name, module in student.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            first_lora_module = module
            lora_B_init = module.lora_B['default'].weight.clone()
            print(f"\n📊 Initial LoRA B statistics:")
            print(f"   Mean: {lora_B_init.mean().item():.6e}")
            print(f"   Std:  {lora_B_init.std().item():.6e}")
            print(f"   Max:  {lora_B_init.abs().max().item():.6e}")
            break
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"Starting ABM Training")
    print(f"{'='*80}\n")
    
    student.train()
    step = 0
    match_acc_log = []
    ab_loss_log = []
    t0 = time.time()
    
    for epoch in range(num_epochs):
        num_layers_count = len(teacher_layers)
        match_totals = [0.0] * num_layers_count
        epoch_ab_loss = 0.0
        total_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            clear_activations()
            
            try:
                # Teacher forward (no grad)
                with torch.no_grad():
                    _ = teacher(**batch)
                
                # Student forward (with grad)
                optimizer.zero_grad()
                _ = student(**batch)
                
                # Check activations
                if len(teacher_acts) == 0 or len(student_acts) == 0:
                    if step == 0:
                        print("❌ No activations captured on first step!")
                    continue
                
                common_keys = set(teacher_acts.keys()) & set(student_acts.keys())
                if len(common_keys) == 0:
                    if step == 0:
                        print("❌ No common keys on first step!")
                    continue
                
                # Compute AB loss
                ab_loss = compute_ab_loss(
                    student_acts, 
                    teacher_acts, 
                    layer_weights, 
                    margin=margin
                )
                
                # Backward
                ab_loss.backward()
                optimizer.step()
                
                # Logging
                epoch_ab_loss += ab_loss.item()
                ab_loss_log.append((step, ab_loss.item()))
                
                # Matching accuracy
                match_acc = compute_matching_accuracy(student_acts, teacher_acts)
                for key, acc in match_acc.items():
                    layer_idx = int(key.split('_')[1])
                    if layer_idx < len(match_totals):
                        match_totals[layer_idx] += acc
                
                total_batches += 1
                
                # Progress logging
                if accelerator.is_local_main_process and step % 10 == 0:
                    print(f"Step {step:4d} | Loss: {ab_loss.item():.6f} | Keys: {len(common_keys)}")
                    wandb.log({
                        "ab_loss": ab_loss.item(),
                        "step": step,
                        "common_keys": len(common_keys)
                    })
                    
                    # Track LoRA changes
                    if first_lora_module is not None and lora_B_init is not None:
                        current_B = first_lora_module.lora_B['default'].weight
                        B_change = (current_B - lora_B_init).abs().max().item()
                        if current_B.grad is not None:
                            grad_norm = current_B.grad.norm().item()
                            print(f"           | LoRA ΔB: {B_change:.6e} | Grad: {grad_norm:.6e}")
                
                step += 1
                
                # Early stopping
                if max_steps is not None and step >= max_steps:
                    print(f"\n🛑 Early stopping at step {step}")
                    break
                    
            except Exception as e:
                print(f"❌ Error at step {step}: {str(e)}")
                continue
        
        if max_steps is not None and step >= max_steps:
            break
        
        # Epoch summary
        if total_batches > 0:
            avg_accs = [100.0 * m / total_batches for m in match_totals]
            avg_ab_loss = epoch_ab_loss / total_batches
            
            if accelerator.is_local_main_process:
                print(f"\n{'─'*80}")
                print(f"Epoch {epoch} Summary | Loss: {avg_ab_loss:.6f} | Time: {time.time()-t0:.1f}s")
                print(f"{'─'*80}")
                for i, acc in enumerate(avg_accs):
                    print(f"  Layer {i:2d}: {acc:.2f}% match")
                
                wandb.log({
                    "epoch": epoch,
                    "avg_ab_loss": avg_ab_loss,
                    **{f"layer_{i}_match_acc": acc for i, acc in enumerate(avg_accs)}
                })
                
                match_acc_log.append((epoch, avg_accs))
    
    # Cleanup hooks
    try:
        remove_hooks(teacher_hooks)
        remove_hooks(student_hooks)
        print("\n🧹 Hooks cleaned up")
    except Exception as e:
        print(f"⚠️  Hook cleanup error: {str(e)}")
    
    # Save model
    save_dir = os.path.join(output_dir, wandb_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n💾 Saving model to: {save_dir}")
    student.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Save logs
    if len(match_acc_log) > 0:
        acc_csv_path = os.path.join(save_dir, "match_accuracy.csv")
        with open(acc_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + [f"layer_{i}" for i in range(len(teacher_layers))])
            for epoch, accs in match_acc_log:
                writer.writerow([epoch] + accs)
    
    if len(ab_loss_log) > 0:
        loss_csv_path = os.path.join(save_dir, "ab_loss.csv")
        with open(loss_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "ab_loss"])
            writer.writerows(ab_loss_log)
        
        # Plot loss curve
        steps, losses = zip(*ab_loss_log)
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, linewidth=2)
        plt.xlabel("Step")
        plt.ylabel("AB Loss")
        plt.title("ABM Initialization Loss Curve")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
        plt.close()
    
    # Final summary
    final_avg_loss = sum([loss for _, loss in ab_loss_log[-10:]]) / min(10, len(ab_loss_log)) if ab_loss_log else 0
    
    print(f"\n{'='*80}")
    print(f"✅ Stage 1 Complete!")
    print(f"{'='*80}")
    print(f"Total steps: {step}")
    print(f"Final avg loss: {final_avg_loss:.6f}")
    print(f"Save directory: {save_dir}")
    
    if first_lora_module is not None and lora_B_init is not None:
        final_B = first_lora_module.lora_B['default'].weight
        final_change = (final_B - lora_B_init).abs().max().item()
        print(f"LoRA B change: {final_change:.6e}")
        
        if len(match_acc_log) > 0:
            initial_avg = sum(match_acc_log[0][1]) / len(match_acc_log[0][1])
            final_avg = sum(match_acc_log[-1][1]) / len(match_acc_log[-1][1])
            print(f"Match accuracy: {initial_avg:.1f}% → {final_avg:.1f}% ({final_avg-initial_avg:+.1f}%)")
    
    print(f"{'='*80}\n")
    
    if accelerator.is_local_main_process:
        wandb.finish()


if __name__ == "__main__":
    Fire(main)
