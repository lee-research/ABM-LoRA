# ABM-LoRA Project Structure

```
ABM-LoRA/
├── README.md                    # Main documentation
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
│
├── src/                         # Core implementation
│   ├── __init__.py             # Package metadata
│   ├── utils.py                # General utilities
│   ├── data.py                 # Dataset loaders
│   ├── evaluation.py           # Evaluation scripts
│   ├── logTrainer.py           # Training logger
│   │
│   └── abm/                    # ABM-LoRA core
│       ├── __init__.py         # ABM module exports
│       ├── utils.py            # ABM utilities (633 lines) ⭐
│       ├── stage1.py           # Stage 1: ABM initialization
│       └── stage2.py           # Stage 2: Fine-tuning
│
├── examples/                    # Quick start & tests
│   ├── README.md               # Examples guide
│   ├── __init__.py
│   ├── quick_start.py          # 5-min demo ⭐
│   └── test_abm_utils.py       # Unit tests ⭐
│
├── scripts/                     # Experiment scripts
    └── run_glue_full.sh        # Run all GLUE tasks
```

## Key Files

### ⭐ Core Implementation
- **`src/abm/utils.py`** (633 lines)
  - Activation storage and key mapping
  - Hook management (register/remove)
  - AB loss computation (Equation 22)
  - T5 layer configuration
  - Model path validation

### ⭐ Training Scripts
- **`src/abm/stage1.py`** - ABM initialization
- **`src/abm/stage2.py`** - Task-specific fine-tuning

### ⭐ Examples & Tests
- **`examples/quick_start.py`** - 5-minute demo
- **`examples/test_abm_utils.py`** - Unit tests

## Statistics

| Category | Count | Lines |
|----------|-------|-------|
| Python files | 13 | ~4,500 |
| Core ABM code | 3 | ~1,500 |
| Examples | 2 | ~600 |
| Documentation | 4 | ~800 |

## Usage

### Quick Start
```bash
# Run tests
python examples/test_abm_utils.py

# Run demo
python examples/quick_start.py

# Full training
python src/abm/stage1.py --dataset_name mrpc
python src/abm/stage2.py --stage1_model_path ./output
```

### Import Structure
```python
# Import ABM utilities
from src.abm import (
    compute_ab_loss,
    register_hooks,
    get_t5_layer_config,
)

# Import general utilities
from src.utils import (
    initialize_text_to_text_model,
    transform_dataset,
)

# Import datasets
from src.data import DATASET_MAP
```
