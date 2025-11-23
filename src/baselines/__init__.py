"""
Baseline Methods for ABM-LoRA

This module provides baseline training methods to create teacher models:
- lora_normal.py: Standard LoRA fine-tuning
- lora_ga.py: LoRA-GA fine-tuning (requires PEFT with LoRA-GA support)

These baselines are used to train teacher models that can then be used
for ABM initialization in Stage 1.
"""
