from .head_pruning import HeadPruner

def prune_transformer_heads(model, prune_ratio=0.2):
    """Entry point for test code."""
    pruner = HeadPruner(prune_ratio=prune_ratio)
    return pruner(model)
