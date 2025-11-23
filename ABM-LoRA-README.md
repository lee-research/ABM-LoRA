# ABM-LoRA: Activation Boundary Matching for Fast Convergence in Low-Rank Adaptation

[![arXiv](https://img.shields.io/badge/arXiv-2411.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2411.XXXXX)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **"ABM-LoRA: Activation Boundary Matching for Fast Convergence in Low-Rank Adaptation"**.

> **Authors:** Dongha Lee*, Jinhee Park*, Minjun Kim, Junseok Kwon (* Equal contribution)  
> **Affiliation:** Graduate School of Artificial Intelligence, Chung-Ang University  
> **Status:** Under review

---

## 📌 Overview

We propose **Activation Boundary Matching for Low-Rank Adaptation (ABM-LoRA)**, a principled initialization strategy that substantially accelerates the convergence of low-rank adapters. While LoRA offers high parameter efficiency, its random initialization restricts gradient updates to a mismatched tangent space, causing significant information loss and hindering early convergence.

### Key Contributions
- **Identifies activation boundary misalignment** as a key cause of slow convergence in LoRA
- **Proposes ABM initialization** that aligns adapter activation boundaries with pretrained models
- **Demonstrates effectiveness** across language (T5) and vision transformers
- **Achieves faster convergence** and higher accuracy on GLUE benchmarks

### Reproducibility Note
All results reported in the paper use lr = 1e-4 for the baseline LoRA setup to ensure direct comparability across methods.
This repository additionally provides ABM-LoRA results under lr = 3e-4, which is its strongest configuration in practice.
To verify fairness, matched-learning-rate experiments (Normal-LoRA: 3e-4 vs ABM-LoRA: 3e-4) are also included, and ABM-LoRA continues to outperform the baseline.


---

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/lee-research/ABM-LoRA.git
cd ABM-LoRA

# Create conda environment
conda create -n abm-lora python=3.8
conda activate abm-lora

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

You can verify the complete pipeline (Initialization → Fine-tuning) in just 3 minutes using our self-contained comparison demo.

```bash
# Run the complete demo pipeline (Normal LoRA vs ABM-LoRA)
python examples/quick_start.py
```

This script automatically sets up a frozen teacher model, runs ABM initialization on a small data subset, and demonstrates the convergence improvement.

---

## 🎓 Training Pipeline

To reproduce the full experiments, follow these steps:

### Step 0: Train Baseline (Teacher Model)

First, train a standard LoRA model to serve as the teacher for ABM initialization.

```bash
python -m src.baselines.lora_normal \
    --dataset_name mrpc \
    --lora_rank 8 \
    --lora_alpha 16 \
    --num_epochs 8 \
    --output_dir ./pretrained_lora
```

**Output:** `./pretrained_lora/baseline=normal_lora_...`

### Step 1: ABM Initialization

Initialize a new Student LoRA model by aligning it with the Teacher model.

```bash
python -m src.abm.stage1 \
    --teacher_lora_path ./pretrained_lora/baseline=normal_lora_... \
    --dataset_name mrpc \
    --lora_rank 8 \
    --max_steps 100 \
    --output_dir ./ab_stage1
```

**Output:** `./ab_stage1/stage1_abm_init_...`

### Step 2: Fine-tuning

Fine-tune the ABM-initialized student model on the target dataset.

```bash
python -m src.abm.stage2 \
    --stage1_model_path ./ab_stage1/stage1_abm_init_... \
    --dataset_name mrpc \
    --num_epochs 8 \
    --learning_rate 3e-4 \
    --output_dir ./ab_stage2
```

**Output:** `./ab_stage2/stage2_finetuning_...`

---

## 📊 Dataset Preparation

### GLUE Benchmark

Datasets are automatically downloaded via HuggingFace `datasets` when running the scripts:

```python
from datasets import load_dataset
dataset = load_dataset("glue", "mrpc")
```

Supported GLUE tasks: MNLI, SST-2, CoLA, QNLI, MRPC, RTE, STSB

---

## 📈 Evaluation

Evaluate the trained model on the validation set:

```bash
python -m src.evaluation \
    --model_path ./ab_stage2/stage2_finetuning_... \
    --dataset_name mrpc \
    --split validation
```

---

## 📁 Project Structure

```
ABM-LoRA/
├── README.md
├── requirements.txt
├── LICENSE
├── src/
│   ├── __init__.py
│   ├── data.py                 # Dataset loading & caching
│   ├── abm/                    # ABM Method Source
│   │   ├── __init__.py
│   │   ├── stage1.py           # Stage 1: ABM Initialization
│   │   ├── stage2.py           # Stage 2: Fine-tuning
│   │   └── utils.py            # Core ABM Logic (Hooks & Loss)
│   ├── baselines/              # Baseline Methods
│   │   ├── lora_normal.py      # Standard LoRA (Teacher Training)
│   │   └── README.md
│   ├── logTrainer.py           # Custom Trainer with logging
│   ├── evaluation.py           # Evaluation script
│   └── utils.py                # General utilities
├── scripts/
│   └── run_glue_full.sh        # Run full pipeline script
└── examples/
    ├── quick_start.py          # 🚀 Comparison Demo
    ├── test_abm_utils.py       # Unit Tests
    └── README.md
```
---

## 🙏 Acknowledgements

This work was supported by the Graduate School of Artificial Intelligence at Chung-Ang University. We thank the authors of LoRA, PEFT, and LoRA-GA for their foundational work.

---

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: ia06073@cau.ac.kr

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.