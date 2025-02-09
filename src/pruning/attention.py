"""Implementation of attention head pruning for transformer models."""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .base import BasePruningMethod


class AttentionHeadPruning(BasePruningMethod):
    def __init__(
        self,
        importance_metric: str = "weight_magnitude",
        layer_pooling: str = "mean",
    ):
        """Initialize attention head pruning.
        
        Args:
            importance_metric: Method to compute head importance
            layer_pooling: How to combine scores across layers
        """
        super().__init__()
        self.importance_metric = importance_metric
        self.layer_pooling = layer_pooling
        
    def compute_importance_scores(self, model: PreTrainedModel) -> Dict[str, torch.Tensor]:
        scores = []
        for name, module in model.named_modules():
            if hasattr(module, "q_proj") and hasattr(module, "num_attention_heads"):
                # Get query weights and reshape to [num_heads, head_dim, hidden_size]
                q_weights = module.q_proj.weight
                head_dim = module.head_dim
                num_heads = module.num_attention_heads
                q_weights = q_weights.view(num_heads, head_dim, -1)
                
                # Compute importance scores (default: mean absolute weight magnitude)
                if self.importance_metric == "weight_magnitude":
                    score = q_weights.abs().mean(dim=(1, 2))
                    
                scores.append((name, score))
                
        return dict(scores)
        
    def select_pruning_targets(
        self,
        importance_scores: Dict[str, torch.Tensor],
        pruning_ratio: float
    ) -> Dict[str, torch.Tensor]:
        total_heads = sum(score.size(0) for score in importance_scores.values())
        num_heads_to_prune = int(total_heads * pruning_ratio)
        
        # Flatten scores across layers
        flat_scores = []
        for name, score in importance_scores.items():
            flat_scores.extend((name, idx, s.item()) 
                             for idx, s in enumerate(score))
            
        # Sort by importance (ascending)
        flat_scores.sort(key=lambda x: x[2])
        
        # Select heads to prune
        targets = {}
        for name, idx, _ in flat_scores[:num_heads_to_prune]:
            if name not in targets:
                targets[name] = []
            targets[name].append(idx)
            
        return {name: torch.tensor(indices) for name, indices in targets.items()}
        
    def apply_pruning(
        self,
        model: PreTrainedModel,
        pruning_targets: Dict[str, torch.Tensor]
    ) -> PreTrainedModel:
        for name, module in model.named_modules():
            if name in pruning_targets:
                heads_to_prune = pruning_targets[name]
                
                # Create mask for remaining heads
                mask = torch.ones(module.num_attention_heads)
                mask[heads_to_prune] = 0
                mask = mask.bool()
                
                # Update attention head count
                remaining_heads = mask.sum().item()
                module.num_attention_heads = remaining_heads
                
                # Prune query, key, value projections
                head_dim = module.head_dim
                hidden_size = module.config.hidden_size
                
                def prune_projection(proj, mask):
                    weights = proj.weight.view(-1, head_dim, hidden_size)
                    weights = weights[mask]
                    proj.weight = nn.Parameter(weights.reshape(-1, hidden_size))
                    if proj.bias is not None:
                        bias = proj.bias.view(-1, head_dim)
                        bias = bias[mask]
                        proj.bias = nn.Parameter(bias.reshape(-1))
                        
                prune_projection(module.q_proj, mask)
                
                # Handle grouped-query attention
                if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
                    kv_mask = mask[::module.num_key_value_groups]
                    module.num_key_value_heads = kv_mask.sum().item()
                    prune_projection(module.k_proj, kv_mask)
                    prune_projection(module.v_proj, kv_mask)
                else:
                    prune_projection(module.k_proj, mask)
                    prune_projection(module.v_proj, mask)
                    
                # Update output projection
                o_weights = module.o_proj.weight.view(hidden_size, -1, head_dim)
                o_weights = o_weights[:, mask]
                module.o_proj.weight = nn.Parameter(o_weights.reshape(hidden_size, -1))
                
        return model
