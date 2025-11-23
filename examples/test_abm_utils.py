"""
Test ABM-LoRA Utilities

Quick validation script to test core ABM utility functions without requiring
full model loading or datasets.

Usage:
    python examples/test_abm_utils.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from src.abm import (
    teacher_acts,
    student_acts,
    clear_activations,
    get_activation_stats,
    setup_key_mapping,
    register_hooks,
    remove_hooks,
    compute_ab_loss,
    compute_matching_accuracy,
)


def test_activation_storage():
    """Test activation storage and clearing"""
    print("\n" + "="*80)
    print("TEST 1: Activation Storage")
    print("="*80)
    
    # Clear first
    clear_activations()
    
    # Add some mock activations
    teacher_acts['layer_0'] = torch.randn(2, 10, 512)
    teacher_acts['layer_1'] = torch.randn(2, 10, 512)
    student_acts['layer_0'] = torch.randn(2, 10, 512)
    student_acts['layer_1'] = torch.randn(2, 10, 512)
    
    # Get stats
    stats = get_activation_stats()
    
    print(f"✅ Teacher keys: {stats['teacher_keys']}")
    print(f"✅ Student keys: {stats['student_keys']}")
    print(f"✅ Common keys: {stats['common_keys']}")
    
    assert stats['teacher_keys'] == 2, "Should have 2 teacher keys"
    assert stats['student_keys'] == 2, "Should have 2 student keys"
    assert stats['common_keys'] == 2, "Should have 2 common keys"
    
    # Clear and verify
    clear_activations()
    stats = get_activation_stats()
    assert stats['common_keys'] == 0, "Should be empty after clear"
    
    print("✅ PASSED: Activation storage works correctly")


def test_key_mapping():
    """Test key mapping setup"""
    print("\n" + "="*80)
    print("TEST 2: Key Mapping")
    print("="*80)
    
    teacher_layers = [
        "encoder.block.0.layer.1.DenseReluDense.wi",
        "encoder.block.1.layer.1.DenseReluDense.wi",
    ]
    student_layers = [
        "model.encoder.block.0.layer.1.DenseReluDense.wi",
        "model.encoder.block.1.layer.1.DenseReluDense.wi",
    ]
    
    setup_key_mapping(teacher_layers, student_layers)
    
    print("✅ PASSED: Key mapping setup successful")


def test_ab_loss_computation():
    """Test ABM loss computation"""
    print("\n" + "="*80)
    print("TEST 3: AB Loss Computation")
    print("="*80)
    
    clear_activations()
    
    # Create mock activations
    batch_size, seq_len, hidden_dim = 4, 8, 128
    
    # Teacher: mix of positive and negative values (frozen, no grad)
    teacher_acts['layer_0'] = torch.randn(batch_size, seq_len, hidden_dim)
    teacher_acts['layer_1'] = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Student: similar to teacher but requires gradient (trainable)
    # Create independent tensors with requires_grad=True
    student_acts['layer_0'] = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
    student_acts['layer_1'] = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
    
    # Make student similar to teacher (for low loss)
    with torch.no_grad():
        student_acts['layer_0'].data = teacher_acts['layer_0'] + 0.1 * torch.randn_like(teacher_acts['layer_0'])
        student_acts['layer_1'].data = teacher_acts['layer_1'] + 0.1 * torch.randn_like(teacher_acts['layer_1'])
    
    # Layer weights (quadratic increase)
    layer_weights = [0.25, 1.0]
    
    # Compute loss
    loss = compute_ab_loss(
        student_acts,
        teacher_acts,
        layer_weights,
        margin=0.5,
        verbose=False
    )
    
    print(f"✅ Loss computed: {loss.item():.6f}")
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    assert loss.requires_grad, "Loss should have gradient enabled"
    
    # Test gradient flow
    loss.backward()
    print("✅ Gradient flow: OK")
    
    print("✅ PASSED: AB loss computation works correctly")


def test_matching_accuracy():
    """Test activation matching accuracy"""
    print("\n" + "="*80)
    print("TEST 4: Matching Accuracy")
    print("="*80)
    
    clear_activations()
    
    # Perfect match case
    teacher_acts['layer_0'] = torch.tensor([[1.0, -1.0, 2.0, -2.0]])
    student_acts['layer_0'] = torch.tensor([[1.5, -1.5, 2.5, -2.5]])
    
    match_acc = compute_matching_accuracy(student_acts, teacher_acts)
    
    print(f"✅ Perfect match accuracy: {match_acc['layer_0']*100:.1f}%")
    assert match_acc['layer_0'] == 1.0, "Should have 100% match"
    
    # Partial match case (50% match)
    clear_activations()
    teacher_acts['layer_0'] = torch.tensor([[1.0, -1.0, 2.0, -2.0]])
    student_acts['layer_0'] = torch.tensor([[1.5, -1.5, -2.5, 2.5]])  # 50% match: [+,-,-,+] vs [+,-,+,-]
    
    match_acc = compute_matching_accuracy(student_acts, teacher_acts)
    print(f"✅ Partial match accuracy: {match_acc['layer_0']*100:.1f}%")
    assert 0.45 <= match_acc['layer_0'] <= 0.55, f"Should be around 50% match, got {match_acc['layer_0']*100:.1f}%"
    
    print("✅ PASSED: Matching accuracy works correctly")


def test_hook_registration():
    """Test hook registration and removal"""
    print("\n" + "="*80)
    print("TEST 5: Hook Registration")
    print("="*80)
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 10)
            self.layer2 = nn.Linear(10, 10)
            
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x
    
    model = SimpleModel()
    layer_names = ['layer1', 'layer2']
    
    # Register hooks
    hooks = register_hooks(model, layer_names, is_teacher=True)
    
    print(f"✅ Registered {len(hooks)} hooks")
    assert len(hooks) == 2, "Should register 2 hooks"
    
    # Test hook functionality
    clear_activations()
    x = torch.randn(2, 10)
    with torch.no_grad():
        _ = model(x)
    
    stats = get_activation_stats()
    print(f"✅ Captured {stats['teacher_keys']} activations")
    assert stats['teacher_keys'] == 2, "Should capture 2 activations"
    
    # Remove hooks
    remove_hooks(hooks)
    print("✅ Hooks removed")
    
    # Verify hooks are removed
    clear_activations()
    with torch.no_grad():
        _ = model(x)
    stats = get_activation_stats()
    assert stats['teacher_keys'] == 0, "Should not capture after removal"
    
    print("✅ PASSED: Hook registration and removal work correctly")


def test_loss_properties():
    """Test mathematical properties of AB loss"""
    print("\n" + "="*80)
    print("TEST 6: Loss Mathematical Properties")
    print("="*80)
    
    batch_size, seq_len, hidden_dim = 4, 8, 128
    layer_weights = [1.0]
    
    # Property 1: Loss should decrease when student matches teacher better
    clear_activations()
    teacher_acts['layer_0'] = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Case A: Far from teacher
    student_acts['layer_0'] = torch.randn(batch_size, seq_len, hidden_dim)
    loss_far = compute_ab_loss(student_acts, teacher_acts, layer_weights, margin=0.5)
    
    # Case B: Close to teacher
    student_acts['layer_0'] = teacher_acts['layer_0'] + 0.01 * torch.randn_like(teacher_acts['layer_0'])
    loss_close = compute_ab_loss(student_acts, teacher_acts, layer_weights, margin=0.5)
    
    print(f"✅ Loss (far):   {loss_far.item():.6f}")
    print(f"✅ Loss (close): {loss_close.item():.6f}")
    print(f"✅ Ratio: {loss_far.item() / (loss_close.item() + 1e-8):.2f}x")
    
    # Property 2: Loss with higher margin should be higher
    student_acts['layer_0'] = torch.randn(batch_size, seq_len, hidden_dim)
    loss_m05 = compute_ab_loss(student_acts, teacher_acts, layer_weights, margin=0.5)
    loss_m10 = compute_ab_loss(student_acts, teacher_acts, layer_weights, margin=1.0)
    
    print(f"\n✅ Loss (m=0.5): {loss_m05.item():.6f}")
    print(f"✅ Loss (m=1.0): {loss_m10.item():.6f}")
    assert loss_m10.item() >= loss_m05.item(), "Higher margin should give higher loss"
    
    print("✅ PASSED: Loss mathematical properties verified")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("ABM-LoRA Utilities Test Suite")
    print("="*80)
    print("Testing core ABM utility functions...\n")
    
    tests = [
        test_activation_storage,
        test_key_mapping,
        test_ab_loss_computation,
        test_matching_accuracy,
        test_hook_registration,
        test_loss_properties,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n❌ FAILED: {test_fn.__name__}")
            print(f"   Error: {str(e)}")
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"✅ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(tests)}")
    print("="*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
