from typing import Dict
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention

from .config import HeadImportanceMetric, PruningConfig

class WeightMagnitudeImportance(HeadImportanceMetric):
    """Compute head importance based on weight magnitudes."""
    
    def __init__(self, config: PruningConfig):
        self.config = config
    
    def __call__(self, model: nn.Module) -> Dict[int, torch.Tensor]:
        """Compute importance scores for attention heads based on weight magnitudes.
        
        Args:
            model: The LlamaForCausalLM model
            
        Returns:
            Dictionary mapping layer indices to head importance scores (shape: [num_heads])
        """
        head_importance = {}
        
        for name, module in model.named_modules():
            if isinstance(module, LlamaAttention):
                # Extract layer index from name (format: model.layers.{idx}.self_attn)
                layer_idx = int(name.split('.')[2])
                
                # Get query weights and compute magnitude
                q_weights = module.q_proj.weight  # [num_heads * head_dim, hidden_size]
                head_dim = model.config.hidden_size // model.config.num_attention_heads
                num_heads = model.config.num_attention_heads
                
                # Reshape to [num_heads, head_dim, hidden_size]
                reshaped_weights = q_weights.view(num_heads, head_dim, -1)
                
                # Compute L2 norm across head dimensions
                importance = torch.norm(reshaped_weights, dim=(1, 2))
                
                head_importance[layer_idx] = importance
        
        return head_importance

def get_importance_metric(config: PruningConfig) -> HeadImportanceMetric:
    """Factory function to get importance metric based on config."""
    if config.importance_metric == "weight_magnitude":
        return WeightMagnitudeImportance(config)
    else:
        raise ValueError(f"Unknown importance metric: {config.importance_metric}")
