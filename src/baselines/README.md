# Baseline Methods

This directory contains baseline training scripts for creating teacher models used in ABM-LoRA Stage 1.

## 📁 Contents

| File | Method | Description |
|------|--------|-------------|
| `lora_normal.py` | **Standard LoRA** | Vanilla LoRA fine-tuning without special initialization |

---

## 🎯 Purpose

These baselines serve two purposes:

1. **Create teacher models** for ABM-LoRA Stage 1 initialization
2. **Provide comparison baselines** for evaluating ABM-LoRA performance

---

## 🚀 Usage

### Standard LoRA Baseline

Train a vanilla LoRA model:

```bash
python -m src.baselines.lora_normal \
    --dataset_name mrpc \
    --lora_rank 8 \
    --lora_alpha 16 \
    --num_epochs 8 \
    --batch_size 32 \
    --learning_rate 1e-4
```

**Output:**
- Trained model saved to `./baseline_normal/baseline=normal_lora_model=t5-base_...`
- Loss history CSV: `loss_history.csv`
- Loss curve plot: `loss_curve.png`
- The folder name always begins with baseline=normal_lora_model= and ends with _seed=[seed].

---

## 📊 Complete Pipeline

### Step 0: Train Baseline (Teacher)

```bash
# Train standard LoRA as teacher
python -m src.baselines.lora_normal \
    --dataset_name mrpc \
    --output_dir ./teachers
```

### Step 1: ABM Initialization (Using Teacher)

```bash
# Initialize with ABM using the teacher
python -m src.abm.stage1 \
    --teacher_lora_path ./teachers/baseline=normal_lora_... \
    --dataset_name mrpc \
    --max_steps 100
```

### Step 2: Fine-tuning

```bash
# Fine-tune the initialized model
python -m src.abm.stage2 \
    --stage1_model_path ./ab_stage1/... \
    --dataset_name mrpc \
    --num_epochs 8
```

---

## ⚙️ Configuration

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_id` | "t5-base" | HuggingFace model ID |
| `lora_rank` | 8 | LoRA rank |
| `lora_alpha` | 16 | LoRA alpha (scaling) |
| `lora_dropout` | 0.05 | LoRA dropout rate |
| `dataset_name` | "mrpc" | GLUE dataset name |
| `num_epochs` | 8 | Training epochs |
| `batch_size` | 32 | Per-device batch size |
| `learning_rate` | 1e-4 | Learning rate |
| `seed` | 5 | Random seed |

### Advanced Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_length` | 128 | Max sequence length |
| `warmup_ratio` | 0.03 | LR warmup ratio |
| `weight_decay` | 0.0 | Weight decay |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `early_stopping_patience` | 5 | Early stopping patience |

---

## 📈 Expected Results

### Standard LoRA (MRPC)

**Output structure:**
```
baseline_normal/
└── baseline=normal_lora_model=t5-base_dataset=mrpc_...
    ├── adapter_config.json
    ├── adapter_model.bin
    ├── loss_history.csv
    ├── loss_curve.png
    └── tokenizer files...
```

## 📝 Notes

- **Teacher quality matters**: Better trained teachers generally lead to better ABM initialization
- **Hyperparameters**: These baselines use the same hyperparameters as reported in the paper
- **Reproducibility**: Set `--seed` for reproducible results

---

## 🔗 See Also

- [Main README](../../ABM-LoRA-README.md) - Full project documentation
- [Examples](../../examples/README.md) - Quick start examples
- [Stage 1](../abm/stage1.py) - ABM initialization
- [Stage 2](../abm/stage2.py) - Fine-tuning
