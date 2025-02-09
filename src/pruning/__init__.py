"""Pruning methods for transformer models."""
from typing import Callable

from .attention import AttentionHeadPruning


def create_pruning_function(
    pruning_ratio: float = 0.2,
    importance_metric: str = "weight_magnitude",
    layer_pooling: str = "mean"
) -> Callable:
    """Create pruning function for test harness.
    
    Args:
        pruning_ratio: Fraction of attention heads to prune
        importance_metric: Method to compute head importance
        layer_pooling: How to combine scores across layers
        
    Returns:
        Callable that takes a model and returns a pruned model
    """
    pruning = AttentionHeadPruning(
        importance_metric=importance_metric,
        layer_pooling=layer_pooling
    )
    
    def prune_model(model):
        return pruning(model, pruning_ratio=pruning_ratio)
        
    return prune_model
