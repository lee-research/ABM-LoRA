"""
ABM (Activation Boundary Matching) module

This module provides the core functionality for ABM-LoRA:
- Stage 1: Activation boundary matching initialization
- Stage 2: Task-specific fine-tuning
- Utilities: Hook management, loss computation, layer configuration
"""

from .utils import (
    # Activation storage
    teacher_acts,
    student_acts,
    clear_activations,
    get_activation_stats,
    
    # Hook management
    register_hooks,
    remove_hooks,
    setup_key_mapping,
    
    # Loss computation
    compute_ab_loss,
    compute_matching_accuracy,
    
    # Layer configuration
    get_t5_layer_config,
    detect_t5_structure,
    validate_model_paths,
)

__all__ = [
    # Activation storage
    "teacher_acts",
    "student_acts",
    "clear_activations",
    "get_activation_stats",
    
    # Hook management
    "register_hooks",
    "remove_hooks",
    "setup_key_mapping",
    
    # Loss computation
    "compute_ab_loss",
    "compute_matching_accuracy",
    
    # Layer configuration
    "get_t5_layer_config",
    "detect_t5_structure",
    "validate_model_paths",
]
