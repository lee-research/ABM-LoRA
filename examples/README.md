# ABM-LoRA Examples

This directory contains example scripts and tests to help you get started with ABM-LoRA.
The examples in this directory are designed to help users reproduce the key behavior of ABM-LoRA without running full experiments.


## 📁 Contents

| File | Description | Runtime |
|------|-------------|---------|
| `test_abm_utils.py` | Unit tests for ABM utility functions | ~10 seconds |
| `quick_start.py` | Complete pipeline demo with minimal data | ~3-5 minutes |

---

## 🧪 Testing Utilities

Run unit tests to verify all ABM functions work correctly:

```bash
python examples/test_abm_utils.py
```

**What it tests:**
- ✅ Activation storage and clearing
- ✅ Key mapping between teacher/student
- ✅ AB loss computation (Equation 22)
- ✅ Matching accuracy calculation
- ✅ Hook registration and removal
- ✅ Loss mathematical properties

**Expected output:**
```
================================================================================
ABM-LoRA Utilities Test Suite
================================================================================
Testing core ABM utility functions...

================================================================================
TEST 1: Activation Storage
================================================================================
✅ Teacher keys: 2
✅ Student keys: 2
✅ Common keys: 2
✅ PASSED: Activation storage works correctly

... (5 more tests)

================================================================================
Test Summary
================================================================================
✅ Passed: 6/6
================================================================================
```

---

## 🚀 Quick Start Demo

Run a complete ABM-LoRA pipeline with minimal data:

```bash
python examples/quick_start.py
```

**What it does:**

### Stage 1: ABM Initialization
1. Loads pretrained T5-base as frozen teacher
2. Initializes T5-base with LoRA adapters (student)
3. Runs activation boundary matching for 50 steps
4. Saves initialized model

### Stage 2 (Suggested)
After running quick start, you can fine-tune with:
```bash
python src/abm/stage2.py \
    --stage1_model_path ./quick_start_output \
    --dataset_name mrpc \
    --num_epochs 8
```

**Expected output:**
```
================================================================================
🚀 ABM-LoRA Quick Start Demo
================================================================================
This demo shows the complete ABM-LoRA pipeline with minimal data.
Expected runtime: 3-5 minutes

📍 Device: cuda
...
✅ Stage 1 complete!
   Initial loss: 0.245617
   Final loss:   0.083421
   Improvement:  66.0%
...
================================================================================
✅ Quick Start Demo Complete!
================================================================================
```

---

## ⚙️ Customization

### Quick Start Options

```bash
# Faster demo (32 samples, 30 steps)
python examples/quick_start.py \
    --sample_size 32 \
    --max_steps 30

# Different dataset
python examples/quick_start.py \
    --dataset_name cola \
    --sample_size 64

# Custom output directory
python examples/quick_start.py \
    --output_dir ./my_experiment
```

### Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_name` | "mrpc" | GLUE dataset name |
| `sample_size` | 64 | Number of samples for demo |
| `num_layers` | 6 | Number of layers for ABM |
| `max_steps` | 50 | ABM initialization steps |
| `margin` | 0.5 | Margin for hinge loss |
| `learning_rate` | 3e-4 | Learning rate for ABM |
| `lora_rank` | 8 | LoRA rank |
| `lora_alpha` | 16 | LoRA alpha |
| `output_dir` | "./quick_start_output" | Save directory |

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Use smaller sample size
python examples/quick_start.py --sample_size 32

# Or use CPU
CUDA_VISIBLE_DEVICES="" python examples/quick_start.py
```

### Import Errors
Make sure you're in the ABM-LoRA root directory:
```bash
cd /path/to/ABM-LoRA
python examples/quick_start.py
```

### Slow Execution
The demo is designed to be fast. If it's slow:
- Reduce `sample_size` (try 32)
- Reduce `max_steps` (try 30)
- Check if you're using GPU

---

## 📊 Understanding the Output

### ABM Loss (Stage 1)

The loss typically:
- Starts around 0.2-0.3
- Decreases to 0.05-0.15
- Shows 40-70% improvement

**Good signs:**
- ✅ Loss steadily decreasing
- ✅ Matching accuracy increasing
- ✅ No NaN or Inf values

**Bad signs:**
- ❌ Loss increasing or fluctuating wildly
- ❌ NaN or Inf values appear
- ❌ Matching accuracy not improving

### Matching Accuracy

Measures how well student activation signs match teacher:
- **< 60%**: Poor initialization (might need more steps)
- **60-80%**: Good initialization
- **> 80%**: Excellent initialization

---

## 🔗 Next Steps

After running these examples:

1. **Full Experiments**: Run complete GLUE experiments
   ```bash
   bash scripts/run_glue_full.sh
   ```

2. **Custom Datasets**: Modify `src/data.py` to add your dataset

3. **Hyperparameter Tuning**: Adjust ABM settings in Stage 1

4. **Paper Reproduction**: See main README for full experiment configs

---

## 📝 Notes

- **Quick start uses tiny data** - Results may not reflect full training performance
- **Test utilities require no external data** - Safe to run anytime
- **Examples are self-contained** - They manage their own imports and paths

---

## 💬 Questions?

If you encounter issues:
1. Check the main [README](../ABM-LoRA-README.md)
2. Verify dependencies: `pip install -r ../requirements.txt`
3. Open an issue on GitHub
