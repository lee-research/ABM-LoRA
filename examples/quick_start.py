"""
ABM-LoRA Quick Start Demo

A minimal example demonstrating the complete ABM-LoRA pipeline:
    Stage 1: Activation Boundary Matching initialization
    Stage 2: Task-specific fine-tuning

This demo uses a tiny subset of data for quick execution (~3-5 minutes).

Usage:
    python examples/quick_start.py
    
    # Or with custom settings:
    python examples/quick_start.py --sample_size 32 --max_steps 50
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from fire import Fire
from peft import LoraConfig, get_peft_model, PeftModel
import warnings
warnings.filterwarnings('ignore')

from src.utils import (
    transform_dataset,
    initialize_text_to_text_model,
    find_all_linear_modules,
)
from src.data import DATASET_MAP
from src.abm import (
    register_hooks,
    remove_hooks,
    compute_ab_loss,
    compute_matching_accuracy,
    get_t5_layer_config,
    clear_activations,
    get_activation_stats,
    validate_model_paths,
    setup_key_mapping,
    teacher_acts,
    student_acts,
)


def main(
    # Data config
    dataset_name: str = "mrpc",
    sample_size: int = 64,  # Small for quick demo
    
    # ABM config
    num_layers: int = 6,
    max_steps: int = 50,  # Quick initialization
    margin: float = 0.5,
    learning_rate: float = 3e-4,
    
    # LoRA config
    lora_rank: int = 8,
    lora_alpha: int = 16,
    
    # Output
    output_dir: str = "./quick_start_output",
):
    """
    Quick start demo for ABM-LoRA.
    
    Demonstrates the complete pipeline with minimal data for fast execution.
    """
    
    print("\n" + "="*80)
    print("🚀 ABM-LoRA Quick Start Demo")
    print("="*80)
    print("This demo shows the complete ABM-LoRA pipeline with minimal data.")
    print("Expected runtime: 3-5 minutes\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 Device: {device}")
    
    # ========================================================================
    # Step 0: Prepare Data
    # ========================================================================
    print("\n" + "-"*80)
    print("📊 Step 0: Loading dataset")
    print("-"*80)
    
    dataset_func = DATASET_MAP[dataset_name]
    train_set, val_set, _ = dataset_func()
    
    print(f"✅ Dataset loaded: {dataset_name}")
    print(f"   Train size: {len(train_set)}")
    print(f"   Val size: {len(val_set)}")
    
    # ========================================================================
    # Step 1: Initialize Models
    # ========================================================================
    print("\n" + "-"*80)
    print("🔧 Step 1: Initializing models")
    print("-"*80)
    
    # Teacher model (frozen pretrained)
    print("Loading teacher model (pretrained T5)...")
    teacher, tokenizer = initialize_text_to_text_model(
        "t5-base",
        "ConditionalGeneration",
        "fp32",
        flash_attention=False
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher = teacher.to(device)
    print("✅ Teacher model loaded and frozen")
    
    # Student model (trainable LoRA)
    print("\nInitializing student model with LoRA...")
    student, _ = initialize_text_to_text_model(
        "t5-base",
        "ConditionalGeneration",
        "fp32",
        flash_attention=False
    )
    
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=find_all_linear_modules(student),
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    student = get_peft_model(student, peft_config)
    student = student.to(device)
    
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student.parameters())
    
    print(f"✅ Student model initialized")
    print(f"   Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Prepare small dataset
    print("\n📦 Preparing dataset...")
    transform_dataset("ConditionalGeneration", tokenizer, train_set, max_length=128)
    
    # Sample tiny subset
    indices = torch.randperm(len(train_set))[:sample_size]
    train_subset = torch.utils.data.Subset(train_set, indices)
    
    dataloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=1,
        shuffle=True
    )
    print(f"✅ Using {len(train_subset)} samples for demo")
    
    # ========================================================================
    # Step 2: Stage 1 - ABM Initialization
    # ========================================================================
    print("\n" + "-"*80)
    print("🎯 Step 2: Stage 1 - Activation Boundary Matching")
    print("-"*80)
    
    # Configure layers
    teacher_layers, student_layers, layer_weights = get_t5_layer_config(
        num_layers=num_layers,
        target_type="wi",
        use_decoder=True,
        teacher_model=teacher,
        student_model=student
    )
    
    # Validate paths
    valid_teacher, _ = validate_model_paths(teacher, teacher_layers, "Teacher")
    valid_student, _ = validate_model_paths(student, student_layers, "Student")
    
    teacher_layers = valid_teacher
    student_layers = valid_student
    layer_weights = layer_weights[:len(teacher_layers)]
    
    setup_key_mapping(teacher_layers, student_layers)
    
    # Register hooks
    print("\n🔗 Registering hooks...")
    teacher_hooks = register_hooks(teacher, teacher_layers, is_teacher=True)
    student_hooks = register_hooks(student, student_layers, is_teacher=False)
    
    # Test hooks
    print("🧪 Testing hooks...")
    test_batch = next(iter(dataloader))
    test_batch = {k: v.to(device) for k, v in test_batch.items()}
    clear_activations()
    
    with torch.no_grad():
        _ = teacher(**test_batch)
    _ = student(**test_batch)
    
    stats = get_activation_stats()
    print(f"✅ Hooks working: {stats['common_keys']} common keys captured")
    
    # ABM training
    print(f"\n🔄 Running ABM initialization ({max_steps} steps)...")
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    
    student.train()
    losses = []
    
    for step, batch in enumerate(dataloader):
        if step >= max_steps:
            break
        
        batch = {k: v.to(device) for k, v in batch.items()}
        clear_activations()
        
        # Teacher forward (no grad)
        with torch.no_grad():
            _ = teacher(**batch)
        
        # Student forward (with grad)
        optimizer.zero_grad()
        _ = student(**batch)
        
        # Compute ABM loss
        ab_loss = compute_ab_loss(
            student_acts,
            teacher_acts,
            layer_weights,
            margin=margin
        )
        
        # Backward
        ab_loss.backward()
        optimizer.step()
        
        losses.append(ab_loss.item())
        
        if (step + 1) % 10 == 0:
            avg_loss = sum(losses[-10:]) / len(losses[-10:])
            match_acc = compute_matching_accuracy(student_acts, teacher_acts)
            avg_match = sum(match_acc.values()) / len(match_acc) if match_acc else 0
            print(f"  Step {step+1:3d}/{max_steps} | Loss: {avg_loss:.6f} | Match: {avg_match*100:.1f}%")
    
    # Cleanup hooks
    remove_hooks(teacher_hooks)
    remove_hooks(student_hooks)
    
    final_loss = sum(losses[-10:]) / len(losses[-10:])
    print(f"\n✅ Stage 1 complete!")
    print(f"   Initial loss: {losses[0]:.6f}")
    print(f"   Final loss:   {final_loss:.6f}")
    print(f"   Improvement:  {(1 - final_loss/losses[0])*100:.1f}%")
    
    # ========================================================================
    # Step 3: Save initialized model
    # ========================================================================
    print("\n" + "-"*80)
    print("💾 Step 3: Saving initialized model")
    print("-"*80)
    
    os.makedirs(output_dir, exist_ok=True)
    student.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Model saved to: {output_dir}")
    
    # ========================================================================
    # Stage 2 would follow here in real training
    # ========================================================================
    print("\n" + "-"*80)
    print("📝 Next Steps")
    print("-"*80)
    print("The initialized model is ready for Stage 2 fine-tuning!")
    print("\nTo run full fine-tuning:")
    print(f"  python src/abm/stage2.py \\")
    print(f"      --stage1_model_path {output_dir} \\")
    print(f"      --dataset_name {dataset_name} \\")
    print(f"      --num_epochs 8")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("✅ Quick Start Demo Complete!")
    print("="*80)
    print("Key Results:")
    print(f"  • Dataset: {dataset_name} ({sample_size} samples)")
    print(f"  • ABM steps: {len(losses)}")
    print(f"  • Loss reduction: {(1 - final_loss/losses[0])*100:.1f}%")
    print(f"  • Model saved: {output_dir}")
    print("\nWhat happened:")
    print("  1. Loaded pretrained T5 as teacher (frozen)")
    print("  2. Initialized student T5 with LoRA adapters")
    print("  3. Ran ABM to align activation boundaries")
    print("  4. Saved initialized model for fine-tuning")
    print("\nThe initialized model should converge faster in Stage 2!")
    print("="*80 + "\n")


if __name__ == "__main__":
    Fire(main)
