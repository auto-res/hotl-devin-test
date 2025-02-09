"""Script for running attention head pruning experiments."""
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.pruning.attention import AttentionHeadPruning


def run_pruning_experiment(
    model_name: str = "HuggingFaceTB/SmolLM-135M",
    pruning_ratio: float = 0.2,
    importance_metric: str = "weight_magnitude",
    layer_pooling: str = "mean",
    dataset_name: Optional[str] = None,
    output_dir: str = "outputs",
    seed: int = 42
):
    """Run attention head pruning experiment.
    
    Args:
        model_name: HuggingFace model identifier
        pruning_ratio: Fraction of attention heads to prune
        importance_metric: Method to compute head importance
        layer_pooling: How to combine scores across layers
        dataset_name: Optional dataset to evaluate on
        output_dir: Directory to save results
        seed: Random seed
    """
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # Initialize pruning method
    pruning = AttentionHeadPruning(
        importance_metric=importance_metric,
        layer_pooling=layer_pooling
    )
    
    # Apply pruning
    pruned_model = pruning(model, pruning_ratio=pruning_ratio)
    
    # Save pruned model
    pruned_model.save_pretrained(f"{output_dir}/pruned_model")
    
    return pruned_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--pruning_ratio", type=float, default=0.2)
    parser.add_argument("--importance_metric", default="weight_magnitude")
    parser.add_argument("--layer_pooling", default="mean")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    run_pruning_experiment(**vars(args))
