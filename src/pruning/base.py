"""Base interface for model pruning methods."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


class BasePruningMethod(ABC):
    """Base class for model pruning methods."""
    
    @abstractmethod
    def compute_importance_scores(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Compute importance scores for prunable components.
        
        Args:
            model: The model to analyze
            
        Returns:
            Dictionary mapping component names to their importance scores
        """
        pass
    
    @abstractmethod
    def select_pruning_targets(
        self, 
        importance_scores: Dict[str, torch.Tensor],
        pruning_ratio: float
    ) -> Dict[str, torch.Tensor]:
        """Select components to prune based on importance scores.
        
        Args:
            importance_scores: Dictionary of importance scores per component
            pruning_ratio: Fraction of components to prune (0.0 to 1.0)
            
        Returns:
            Dictionary mapping component names to tensors indicating which
            parts should be pruned
        """
        pass
    
    @abstractmethod
    def apply_pruning(
        self,
        model: nn.Module,
        pruning_targets: Dict[str, torch.Tensor]
    ) -> nn.Module:
        """Apply pruning to the model.
        
        Args:
            model: The model to prune
            pruning_targets: Dictionary indicating which components to prune
            
        Returns:
            The pruned model
        """
        pass

    def __call__(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.2,
    ) -> nn.Module:
        """Execute pruning pipeline.
        
        Args:
            model: The model to prune
            pruning_ratio: Fraction of components to prune (0.0 to 1.0)
            
        Returns:
            The pruned model
        """
        scores = self.compute_importance_scores(model)
        targets = self.select_pruning_targets(scores, pruning_ratio)
        return self.apply_pruning(model, targets)
