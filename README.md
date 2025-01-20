# LoRANorm: Low-Rank Adaptation with Weight Normalization

This repository implements LoRANorm, a method that combines LoRA (Low-Rank Adaptation) with weight normalization for improved fine-tuning of large language models.

## Installation

```bash
pip install -e .
```

## Usage

### Running Transformer Experiments

The `transformer_experiment.py` script allows you to fine-tune transformer models using LoRANorm. Example usage:

```bash
python experiment/transformer_experiment.py \
    --model-name distilgpt2 \
    --rank 8 \
    --dropout 0.1 \
    --alpha 1.0 \
    --lr 1e-5 \
    --batch-size 4 \
    --seq-length 64 \
    --num-epochs 5
```

### Arguments

- `--model-name`: HuggingFace model name (default: distilgpt2)
- `--rank`: LoRANorm rank for low-rank approximation (default: 4)
- `--dropout`: Dropout probability for LoRANorm (default: 0.1)
- `--alpha`: LoRA scaling factor (default: 1.0)
- `--lr`: Learning rate (default: 5e-5)
- `--batch-size`: Batch size (default: 4)
- `--seq-length`: Maximum sequence length (default: 32)
- `--num-epochs`: Number of training epochs (default: 5)
- `--device`: Device to use for training ('cpu' or 'cuda', default: 'cpu')

### Supported Models

The experiment script supports any causal language model from HuggingFace's model hub. Some recommended small models:

- distilgpt2
- gpt2
- EleutherAI/pythia-70m
- facebook/opt-125m

## Implementation Details

LoRANorm applies weight normalization to LoRA layers for improved training stability and convergence. The implementation:

1. Targets attention and feed-forward layers in transformer models
2. Applies low-rank adaptation with configurable rank
3. Uses weight normalization for better optimization
4. Supports dropout for regularization

For more details, see the docstrings in `loranorm/loranorm.py` and `experiment/transformer_experiment.py`.
