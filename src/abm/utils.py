"""
ABM-LoRA Utilities

Core utilities for Activation Boundary Matching (ABM) initialization.
Implements the loss function from Equation (22) in the paper:

    L_ABM = (1/N) Σ_i Σ_l w_l^2 [max(0, -τ_{i,l} z_{i,l} + m)]^2

where:
    - τ_{i,l} = sgn(z^pt_{i,l}): pretrained activation signs
    - z_{i,l}: student pre-activations
    - w_l: layer-specific weights
    - m: margin parameter
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict


# ============================================================================
# Global Activation Storage
# ============================================================================

teacher_acts: Dict[str, torch.Tensor] = {}
"""Dictionary storing teacher model activations (frozen)"""

student_acts: Dict[str, torch.Tensor] = {}
"""Dictionary storing student model activations (trainable)"""

teacher_to_standard: Dict[str, str] = {}
"""Mapping from teacher layer paths to standardized keys"""

student_to_standard: Dict[str, str] = {}
"""Mapping from student layer paths to standardized keys"""


# ============================================================================
# Key Mapping Setup
# ============================================================================

def setup_key_mapping(
    teacher_layers: List[str],
    student_layers: List[str]
) -> None:
    """
    Setup key mapping between actual model paths and standardized keys.
    
    This allows teacher and student models with different path structures
    (e.g., "encoder.block.0..." vs "model.encoder.block.0...") to be matched
    using standardized keys like "layer_0", "layer_1", etc.
    
    Args:
        teacher_layers: List of teacher model layer paths
        student_layers: List of student model layer paths
        
    Raises:
        Warning if layer counts don't match
    """
    global teacher_to_standard, student_to_standard
    
    teacher_to_standard.clear()
    student_to_standard.clear()
    
    if len(teacher_layers) != len(student_layers):
        print(f"⚠️  Warning: Layer count mismatch - "
              f"Teacher: {len(teacher_layers)}, Student: {len(student_layers)}")
    
    for i, (t_layer, s_layer) in enumerate(zip(teacher_layers, student_layers)):
        standard_key = f"layer_{i}"
        teacher_to_standard[t_layer] = standard_key
        student_to_standard[s_layer] = standard_key
    
    print(f"✅ Key mapping setup: {len(teacher_to_standard)} teacher, "
          f"{len(student_to_standard)} student layers")


# ============================================================================
# Activation Storage Management
# ============================================================================

def clear_activations() -> None:
    """Clear all stored activations."""
    global teacher_acts, student_acts
    teacher_acts.clear()
    student_acts.clear()


def get_activation_stats() -> Dict[str, int]:
    """
    Get statistics about stored activations.
    
    Returns:
        Dictionary containing:
            - teacher_keys: number of teacher activation keys
            - student_keys: number of student activation keys  
            - common_keys: number of matching keys between teacher and student
    """
    stats = {
        "teacher_keys": len(teacher_acts),
        "student_keys": len(student_acts),
        "common_keys": len(set(teacher_acts.keys()) & set(student_acts.keys()))
    }
    return stats


# ============================================================================
# Forward Hook Functions
# ============================================================================

def get_activation_saver(name: str, is_teacher: bool = True):
    """
    Create a forward hook function that saves layer activations.
    
    Args:
        name: Original layer path in the model
        is_teacher: Whether this hook is for teacher (True) or student (False)
        
    Returns:
        Hook function that stores activations in global dictionaries
    """
    def hook_fn(module, input, output):
        try:
            if is_teacher:
                standard_key = teacher_to_standard.get(name, name)
                if isinstance(output, torch.Tensor):
                    teacher_acts[standard_key] = output.detach().float()
            else:
                standard_key = student_to_standard.get(name, name)
                if isinstance(output, torch.Tensor):
                    student_acts[standard_key] = output.float()
                    
        except Exception as e:
            model_type = "Teacher" if is_teacher else "Student"
            print(f"❌ Error in {model_type} hook {name}: {str(e)}")
            
    return hook_fn


def register_hooks(
    model: nn.Module,
    layer_names: List[str],
    is_teacher: bool = True
) -> List:
    """
    Register forward hooks to capture layer outputs.
    
    Args:
        model: PyTorch model
        layer_names: List of layer paths to hook
        is_teacher: Whether this is teacher (True) or student (False) model
        
    Returns:
        List of registered hook handles
        
    Example:
        >>> hooks = register_hooks(model, ["encoder.block.0.layer.1.DenseReluDense.wi"])
        >>> # Run forward pass, activations are stored in teacher_acts/student_acts
        >>> remove_hooks(hooks)
    """
    hooks = []
    successful = 0
    failed_names = []
    
    model_type = "Teacher" if is_teacher else "Student"
    
    for name in layer_names:
        try:
            # Navigate to target module
            module = model
            for attr in name.split('.'):
                if hasattr(module, attr):
                    module = getattr(module, attr)
                else:
                    raise AttributeError(f"Module '{attr}' not found in path '{name}'")
            
            # Register hook
            hook_fn = get_activation_saver(name, is_teacher=is_teacher)
            h = module.register_forward_hook(hook_fn)
            hooks.append(h)
            successful += 1
            
        except (AttributeError, Exception) as e:
            failed_names.append(name)
            if successful < 3:  # Only print first few failures
                print(f"❌ Failed to register hook for {name}: {e}")
    
    print(f"✅ {model_type}: {successful}/{len(layer_names)} hooks registered")
    
    return hooks


def remove_hooks(hooks: List) -> None:
    """
    Remove all registered hooks.
    
    Args:
        hooks: List of hook handles returned by register_hooks()
    """
    removed = 0
    for h in hooks:
        try:
            h.remove()
            removed += 1
        except Exception as e:
            print(f"⚠️  Error removing hook: {e}")
    
    if removed > 0:
        print(f"🧹 Removed {removed}/{len(hooks)} hooks")


# ============================================================================
# ABM Loss Computation (Equation 22)
# ============================================================================

def compute_ab_loss(
    student_acts: Dict[str, torch.Tensor],
    teacher_acts: Dict[str, torch.Tensor],
    layer_weights: List[float],
    margin: float = 1.0,
    verbose: bool = False
) -> torch.Tensor:
    """
    Compute Activation Boundary Matching loss (Equation 22 from paper).
    
    Implements:
        L_ABM = (1/N) Σ_i Σ_l w_l^2 [max(0, -τ_{i,l} z_{i,l} + m)]^2
    
    where:
        - τ_{i,l} = sgn(z^pt_{i,l}): teacher activation signs (binary mask)
        - z_{i,l}: student pre-activations
        - w_l: layer-specific weight (higher for deeper layers)
        - m: margin parameter
    
    The loss encourages student activations to match teacher activation boundaries:
        - If teacher neuron is active (τ > 0): push student activation above margin
        - If teacher neuron is inactive (τ ≤ 0): push student activation below -margin
    
    Args:
        student_acts: Dictionary of student pre-activations {layer_key: tensor}
        teacher_acts: Dictionary of teacher pre-activations {layer_key: tensor}
        layer_weights: List of per-layer weights w_l
        margin: Margin parameter m (default: 1.0)
        verbose: If True, print detailed layer-wise statistics
        
    Returns:
        Scalar loss tensor with gradient enabled
        
    Example:
        >>> loss = compute_ab_loss(student_acts, teacher_acts, [0.25, 0.50, 1.0], margin=0.5)
        >>> loss.backward()
    """
    # Input validation
    if len(student_acts) == 0 or len(teacher_acts) == 0:
        print("❌ Empty activation dictionaries!")
        return torch.tensor(0.0, requires_grad=True)
    
    if len(layer_weights) == 0:
        print("❌ Empty layer weights!")
        return torch.tensor(0.0, requires_grad=True)
    
    total_loss = None  # Will be initialized with first layer loss
    layer_count = 0
    
    # Iterate over standardized keys (layer_0, layer_1, ...)
    for i in range(len(layer_weights)):
        layer_key = f"layer_{i}"
        
        # Check if activations exist
        if layer_key not in student_acts or layer_key not in teacher_acts:
            continue
            
        s = student_acts[layer_key]  # Student pre-activation
        t = teacher_acts[layer_key]  # Teacher pre-activation
        
        # Shape and device validation
        if s.shape != t.shape:
            print(f"⚠️  Shape mismatch at {layer_key}: Student {s.shape} vs Teacher {t.shape}")
            continue
            
        if s.device != t.device:
            t = t.to(s.device)
        
        try:
            # Compute activation boundary matching loss
            # τ = sgn(t): binary mask indicating teacher activation state
            # For ReLU: τ > 0 means neuron is active, τ ≤ 0 means inactive
            
            # Loss for neurons that should be active (t > 0):
            # Penalize if s < margin
            loss_pos = ((s - margin)**2 * ((s < margin) & (t > 0)).float())
            
            # Loss for neurons that should be inactive (t ≤ 0):
            # Penalize if s > -margin  
            loss_neg = ((s + margin)**2 * ((s > -margin) & (t <= 0)).float())
            
            layer_loss = (loss_pos + loss_neg).mean()
            
            # NaN/Inf check
            if torch.isnan(layer_loss) or torch.isinf(layer_loss):
                print(f"⚠️  Invalid loss at {layer_key}: {layer_loss.item()}")
                continue
            
            # Apply layer-specific weight w_l
            weighted_loss = layer_weights[i] * layer_loss
            
            # Initialize or accumulate total_loss
            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss
            
            layer_count += 1
            
            # Verbose logging
            if verbose and i % 4 == 0:
                pos_match = ((s > 0) == (t > 0)).float().mean().item()
                print(f"  Layer {i:2d} | Loss: {layer_loss.item():.6f} | "
                      f"Match: {pos_match*100:.1f}% | Weight: {layer_weights[i]:.3f}")
                
        except Exception as e:
            print(f"❌ Error at {layer_key}: {str(e)}")
            continue
    
    if layer_count == 0 or total_loss is None:
        print("⚠️  No valid layers processed!")
        return torch.tensor(0.0, requires_grad=True)
    
    # Return average loss over layers
    return total_loss / layer_count


def compute_matching_accuracy(
    student_acts: Dict[str, torch.Tensor],
    teacher_acts: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute activation boundary matching accuracy.
    
    Measures how well student activation signs match teacher activation signs:
        accuracy = mean(sgn(student) == sgn(teacher))
    
    Args:
        student_acts: Student activations
        teacher_acts: Teacher activations
        
    Returns:
        Dictionary mapping layer keys to matching accuracies (0.0 to 1.0)
    """
    match_acc = {}
    
    common_keys = set(student_acts.keys()) & set(teacher_acts.keys())
    
    for key in common_keys:
        s = student_acts[key]
        t = teacher_acts[key]
        
        if s.shape != t.shape:
            continue
            
        if s.device != t.device:
            t = t.to(s.device)
        
        try:
            # Compare activation signs
            match = ((s > 0) == (t > 0)).float().mean().item()
            match_acc[key] = match
        except:
            continue
    
    return match_acc


# ============================================================================
# T5 Model Structure Detection
# ============================================================================

def detect_t5_structure(
    model: nn.Module,
    model_name: str = "Model"
) -> Optional[str]:
    """
    Automatically detect T5 model structure pattern.
    
    T5 models can have different path structures depending on how they're wrapped:
        - "encoder.block.{}.layer.1.DenseReluDense.{}"  (base model)
        - "model.encoder.block.{}.layer.1.DenseReluDense.{}"  (PEFT wrapped)
        - "base_model.model.encoder.block.{}.layer.1.DenseReluDense.{}"  (nested PEFT)
    
    Args:
        model: T5 model instance
        model_name: Name for logging purposes
        
    Returns:
        Pattern string with {} placeholders for layer index and target type,
        or None if structure cannot be detected
        
    Example:
        >>> pattern = detect_t5_structure(model)
        >>> layer_path = pattern.format(0, "wi")  # "encoder.block.0.layer.1.DenseReluDense.wi"
    """
    patterns = [
        "encoder.block.{}.layer.1.DenseReluDense.{}",
        "model.encoder.block.{}.layer.1.DenseReluDense.{}",
        "base_model.model.encoder.block.{}.layer.1.DenseReluDense.{}",
    ]
    
    for pattern in patterns:
        try:
            # Test pattern with layer 0, target "wi"
            test_path = pattern.format(0, "wi")
            module = model
            for attr in test_path.split('.'):
                module = getattr(module, attr)
            
            # Check if decoder exists
            decoder_pattern = pattern.replace("encoder.block.{}.layer.1", 
                                             "decoder.block.{}.layer.2")
            decoder_path = decoder_pattern.format(0, "wi")
            decoder_module = model
            try:
                for attr in decoder_path.split('.'):
                    decoder_module = getattr(decoder_module, attr)
                has_decoder = True
            except:
                has_decoder = False
            
            print(f"✅ {model_name} structure detected: {pattern}")
            print(f"   Decoder: {'✅' if has_decoder else '❌'}")
            return pattern
            
        except AttributeError:
            continue
    
    print(f"❌ Could not detect {model_name} T5 structure!")
    return None


# ============================================================================
# T5 Layer Configuration
# ============================================================================

def get_t5_layer_config(
    num_layers: int = 12,
    target_type: str = "wi",
    use_decoder: bool = False,
    teacher_model: Optional[nn.Module] = None,
    student_model: Optional[nn.Module] = None
) -> Tuple[List[str], List[str], List[float]]:
    """
    Get layer configuration for T5 ABM initialization.
    
    Generates layer paths and weights for activation boundary matching.
    Supports both 6-layer (last 6 layers) and 12-layer (all layers) configurations.
    
    Layer weights follow quadratic increase (Equation 22):
        w_l = ((l+1) / L)^2
    where l is the relative layer index and L is total layers per component.
    
    Args:
        num_layers: Number of layers to target (6 or 12)
        target_type: Target sublayer type ("wi" for DenseReluDense input)
        use_decoder: Whether to include decoder layers
        teacher_model: Teacher model for structure detection
        student_model: Student model for structure detection
        
    Returns:
        Tuple containing:
            - teacher_layers: List of teacher layer paths
            - student_layers: List of student layer paths  
            - layer_weights: List of layer-specific weights
            
    Raises:
        ValueError: If num_layers not in [6, 12]
        
    Example:
        >>> teacher_layers, student_layers, weights = get_t5_layer_config(
        ...     num_layers=6, 
        ...     use_decoder=True,
        ...     teacher_model=teacher,
        ...     student_model=student
        ... )
        >>> print(f"Target layers: {len(teacher_layers)}")  # 12 (6 encoder + 6 decoder)
    """
    # Input validation
    if num_layers not in [6, 12]:
        raise ValueError(f"Only num_layers=6 or 12 supported, got: {num_layers}")
    
    valid_targets = ["wi", "wo"]
    if target_type not in valid_targets:
        print(f"⚠️  Warning: target_type '{target_type}' not in {valid_targets}")
    
    # Detect model structures
    teacher_pattern = None
    student_pattern = None
    
    if teacher_model is not None:
        teacher_pattern = detect_t5_structure(teacher_model, "Teacher")
    
    if student_model is not None:
        student_pattern = detect_t5_structure(student_model, "Student")
    
    # Fallback to default patterns
    if teacher_pattern is None:
        print("⚠️  Using default teacher pattern")
        teacher_pattern = "model.encoder.block.{}.layer.1.DenseReluDense.{}"
    
    if student_pattern is None:
        print("⚠️  Using default student pattern")
        student_pattern = "model.encoder.block.{}.layer.1.DenseReluDense.{}"
    
    # Determine layer indices
    if num_layers == 12:
        # All 12 layers: 0,1,2,...,11
        layer_indices = list(range(12))
        print(f"🎯 Using all 12 layers")
    elif num_layers == 6:
        # Last 6 layers: 6,7,8,9,10,11
        layer_indices = list(range(6, 12))
        print(f"🎯 Using last 6 layers: {layer_indices}")
    
    # Generate layer paths
    teacher_layers = []
    student_layers = []
    
    # Encoder layers
    for i in layer_indices:
        teacher_layers.append(teacher_pattern.format(i, target_type))
        student_layers.append(student_pattern.format(i, target_type))
    
    # Decoder layers (if enabled)
    if use_decoder:
        # T5 decoder has DenseReluDense in layer.2 (not layer.1)
        decoder_teacher_pattern = teacher_pattern.replace(
            "encoder.block.{}.layer.1", 
            "decoder.block.{}.layer.2"
        )
        decoder_student_pattern = student_pattern.replace(
            "encoder.block.{}.layer.1",
            "decoder.block.{}.layer.2"
        )
        
        for i in layer_indices:
            teacher_layers.append(decoder_teacher_pattern.format(i, target_type))
            student_layers.append(decoder_student_pattern.format(i, target_type))
        
        print(f"✅ Added decoder layers (layer.2)")
    
    # Compute layer weights: w_l = ((l+1) / L)^2
    total_layers = len(teacher_layers)
    layer_weights = []
    
    if use_decoder:
        # Encoder + Decoder: independent weight increase for each component
        # Encoder weights
        for idx in range(len(layer_indices)):
            encoder_weight = ((idx + 1) / len(layer_indices))**2
            layer_weights.append(encoder_weight)
        
        # Decoder weights
        for idx in range(len(layer_indices)):
            decoder_weight = ((idx + 1) / len(layer_indices))**2
            layer_weights.append(decoder_weight)
        
        print(f"🎯 Weight Distribution:")
        print(f"   Encoder: {layer_weights[0]:.3f} → {layer_weights[len(layer_indices)-1]:.3f}")
        print(f"   Decoder: {layer_weights[len(layer_indices)]:.3f} → {layer_weights[-1]:.3f}")
        
    else:
        # Encoder only
        for idx in range(len(layer_indices)):
            encoder_weight = ((idx + 1) / len(layer_indices))**2
            layer_weights.append(encoder_weight)
        
        print(f"🎯 Weight Distribution:")
        print(f"   Encoder: {layer_weights[0]:.3f} → {layer_weights[-1]:.3f}")
    
    # Summary
    print(f"\n📋 T5 Layer Config Summary:")
    print(f"   Total layers: {total_layers}")
    print(f"   Target type: {target_type}")
    print(f"   Use decoder: {use_decoder}")
    print(f"   Layer indices: {layer_indices}")
    
    # Setup key mapping
    setup_key_mapping(teacher_layers, student_layers)
    
    return teacher_layers, student_layers, layer_weights


# ============================================================================
# Model Path Validation
# ============================================================================

def validate_model_paths(
    model: nn.Module,
    layer_names: List[str],
    model_type: str = "Model"
) -> Tuple[List[str], List[str]]:
    """
    Validate that layer paths exist in the model.
    
    Args:
        model: PyTorch model to validate
        layer_names: List of layer paths to check
        model_type: Name for logging purposes
        
    Returns:
        Tuple containing:
            - valid_paths: List of paths that exist in model
            - invalid_paths: List of paths that don't exist
            
    Example:
        >>> valid, invalid = validate_model_paths(model, layer_list, "Teacher")
        >>> print(f"Valid: {len(valid)}, Invalid: {len(invalid)}")
    """
    valid_paths = []
    invalid_paths = []
    
    for name in layer_names:
        try:
            module = model
            for attr in name.split('.'):
                module = getattr(module, attr)
            valid_paths.append(name)
        except AttributeError:
            invalid_paths.append(name)
    
    print(f"📋 {model_type} Path Validation:")
    print(f"   Valid: {len(valid_paths)}/{len(layer_names)}")
    
    if invalid_paths and len(invalid_paths) <= 3:
        print(f"   Invalid: {invalid_paths}")
    elif invalid_paths:
        print(f"   Invalid: {len(invalid_paths)} paths (showing first 3)")
        print(f"            {invalid_paths[:3]}")
    
    return valid_paths, invalid_paths
