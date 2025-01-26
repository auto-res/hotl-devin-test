# Wide Residual Networks Implementation and Improvements

This repository contains an implementation of Wide Residual Networks (WRN) with three accuracy improvements, focusing on the WRN-28-10 architecture for CIFAR-100 classification.

## Project Structure

```
.
├── src/                # Source code for model implementations
│   ├── __init__.py
│   ├── wrn.py         # Base WRN implementation
│   └── improvements/  # Accuracy improvement implementations
├── experiment/         # Training and evaluation scripts
│   ├── __init__.py
│   └── train_wrn.py   # Main training script
├── requirements.txt    # Project dependencies
└── setup.py           # Package installation
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- tqdm

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training with reduced dataset (for testing):
```bash
python experiment/train_wrn.py --model baseline --epochs 2 --subset 500
```

3. Run full training:
```bash
python experiment/train_wrn.py --model baseline --epochs 200
```

## Model Variations

1. Baseline: WRN-28-10
2. Improvement 1: Enhanced Dropout
3. Improvement 2: Stochastic Depth
4. Improvement 3: Advanced Data Augmentation

## License

MIT License
