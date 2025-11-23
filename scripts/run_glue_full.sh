#!/bin/bash
# ABM-LoRA GLUE Benchmark - Full Pipeline
# This script runs both Stage 1 (ABM initialization) and Stage 2 (fine-tuning)

set -e  # Exit on error

# Configuration
MODEL="t5-base"
DATASET="mrpc"
LORA_RANK=8
LORA_ALPHA=16
SAMPLE_SIZE=128
SEED=5

# Stage 1 config
STAGE1_STEPS=100
STAGE1_LR=3e-4
NUM_LAYERS=6
MARGIN=0.5

# Stage 2 config
STAGE2_EPOCHS=8
STAGE2_LR=3e-4
BATCH_SIZE=32

# Paths
TEACHER_PATH="./pretrained_lora/${MODEL}_${DATASET}"  # Update with your teacher path
STAGE1_OUTPUT="./ab_stage1"
STAGE2_OUTPUT="./ab_stage2"

echo "======================================================================"
echo "ABM-LoRA Pipeline: ${MODEL} on ${DATASET}"
echo "======================================================================"
echo "Configuration:"
echo "  Model: ${MODEL}"
echo "  Dataset: ${DATASET}"
echo "  LoRA Rank: ${LORA_RANK}"
echo "  LoRA Alpha: ${LORA_ALPHA}"
echo "======================================================================"

# Check if teacher model exists
if [ ! -d "${TEACHER_PATH}" ]; then
    echo "❌ Error: Teacher model not found at ${TEACHER_PATH}"
    echo "Please train a baseline LoRA model first or update TEACHER_PATH"
    exit 1
fi

stage 1: ABM Initialization
echo ""
echo "======================================================================"
echo "Stage 1: Activation Boundary Matching Initialization"
echo "======================================================================"


python -m src.abm.stage1 \
    --model_id ${MODEL} \
    --teacher_lora_path ${TEACHER_PATH} \
    --dataset_name ${DATASET} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --sample_size ${SAMPLE_SIZE} \
    --max_steps ${STAGE1_STEPS} \
    --learning_rate ${STAGE1_LR} \
    --num_layers ${NUM_LAYERS} \
    --margin ${MARGIN} \
    --seed ${SEED} \
    --output_dir ${STAGE1_OUTPUT} \
    --wandb_mode offline

# Find the latest stage1 checkpoint
STAGE1_CHECKPOINT=$(ls -t ${STAGE1_OUTPUT} | head -1)
STAGE1_PATH="${STAGE1_OUTPUT}/${STAGE1_CHECKPOINT}"

echo ""
echo "✅ Stage 1 completed. Checkpoint: ${STAGE1_PATH}"

# Stage 2: Fine-tuning
echo ""
echo "======================================================================"
echo "Stage 2: Fine-tuning"
echo "======================================================================"


python -m src.abm.stage2 \
    --model_id ${MODEL} \
    --stage1_model_path ${STAGE1_PATH} \
    --dataset_name ${DATASET} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --num_epochs ${STAGE2_EPOCHS} \
    --learning_rate ${STAGE2_LR} \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED} \
    --output_dir ${STAGE2_OUTPUT} \
    --wandb_mode offline

# Find the latest stage2 checkpoint
STAGE2_CHECKPOINT=$(ls -t ${STAGE2_OUTPUT} | head -1)
STAGE2_PATH="${STAGE2_OUTPUT}/${STAGE2_CHECKPOINT}"

echo ""
echo "======================================================================"
echo "✅ ABM-LoRA Pipeline Complete!"
echo "======================================================================"
echo "Stage 2 checkpoint: ${STAGE2_PATH}"
echo ""
echo "Run evaluation:"
echo "python -m src.evaluation --model_path ${STAGE2_PATH} --dataset_name ${DATASET}"
echo "======================================================================"
