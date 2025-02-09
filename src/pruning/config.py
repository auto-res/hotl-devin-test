from dataclasses import dataclass
from typing import Optional, Callable, Union
import torch

@dataclass
class PruningConfig:
    """Configuration for attention head pruning.
    
    Args:
        pruning_rate: Percentage of heads to prune (between 0 and 1)
        importance_metric: Method to compute head importance
        min_heads: Minimum number of heads to keep per layer
        maintain_gqa_ratio: Whether to maintain grouped-query attention ratio
        random_seed: Random seed for reproducibility
    """
    pruning_rate: float = 0.2
    importance_metric: str = "weight_magnitude"
    min_heads: int = 1
    maintain_gqa_ratio: bool = True
    random_seed: Optional[int] = None

    def __post_init__(self):
        if not 0 <= self.pruning_rate <= 1:
            raise ValueError("pruning_rate must be between 0 and 1")
        if self.min_heads < 1:
            raise ValueError("min_heads must be at least 1")
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)

class HeadImportanceMetric:
    """Base class for head importance metrics."""
    
    def __call__(self, model: torch.nn.Module) -> dict[int, torch.Tensor]:
        """Compute importance scores for attention heads.
        
        Args:
            model: The model to analyze
            
        Returns:
            Dictionary mapping layer indices to head importance scores
        """
        raise NotImplementedError
