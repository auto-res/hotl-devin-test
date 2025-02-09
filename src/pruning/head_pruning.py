import torch
import torch.nn as nn
from transformers import PreTrainedModel
from typing import Dict, List, Optional, Tuple, Union

class HeadPruner:
    def __init__(
        self,
        prune_ratio: float = 0.2,
        importance_metric: str = "weight_magnitude"
    ):
        self.prune_ratio = prune_ratio
        self.importance_metric = importance_metric
    
    def compute_head_importance(
        self,
        model: PreTrainedModel
    ) -> Dict[int, torch.Tensor]:
        """Compute importance scores for each attention head."""
        head_importance = {}
        for name, module in model.named_modules():
            if "self_attn" in name:
                layer_idx = int(name.split('.')[2])
                # Get query weights for importance calculation
                q_weight = module.q_proj.weight
                head_size = module.head_dim
                num_heads = module.num_attention_heads
                
                # Reshape weights to [num_heads, head_size, hidden_size]
                q_weight = q_weight.view(num_heads, head_size, -1)
                
                # Compute importance as average magnitude
                importance = q_weight.abs().mean(dim=(1,2))
                head_importance[layer_idx] = importance
                
        return head_importance
    
    def select_heads_to_prune(
        self,
        head_importance: Dict[int, torch.Tensor]
    ) -> Dict[int, List[int]]:
        """Select least important heads to prune."""
        all_scores = []
        for layer_idx, scores in head_importance.items():
            for head_idx, score in enumerate(scores):
                all_scores.append((layer_idx, head_idx, score.item()))
        
        # Sort by importance score
        all_scores.sort(key=lambda x: x[2])
        
        # Select heads to prune
        num_to_prune = int(len(all_scores) * self.prune_ratio)
        heads_to_prune = {}
        
        for layer_idx, head_idx, _ in all_scores[:num_to_prune]:
            if layer_idx not in heads_to_prune:
                heads_to_prune[layer_idx] = []
            heads_to_prune[layer_idx].append(head_idx)
            
        return heads_to_prune
    
    def prune_heads(
        self,
        model: PreTrainedModel,
        heads_to_prune: Dict[int, List[int]]
    ) -> PreTrainedModel:
        """Modify model architecture to remove selected heads."""
        for layer_idx, heads in heads_to_prune.items():
            layer = model.model.layers[layer_idx]
            attn = layer.self_attn
            
            # Create head mask
            mask = torch.ones(attn.num_attention_heads)
            mask[heads] = 0
            
            # Update attention parameters
            head_size = attn.head_dim
            hidden_size = attn.num_attention_heads * head_size
            new_num_heads = attn.num_attention_heads - len(heads)
            
            # Prune q_proj
            q_weight = attn.q_proj.weight.view(attn.num_attention_heads, head_size, -1)
            q_weight = q_weight[mask.bool()]
            attn.q_proj.weight = nn.Parameter(q_weight.reshape(new_num_heads * head_size, -1))
            
            # Handle grouped-query attention for k_proj and v_proj
            if hasattr(attn, 'num_key_value_groups'):
                kv_mask = mask[::attn.num_key_value_groups]
                new_kv_heads = (new_num_heads + attn.num_key_value_groups - 1) // attn.num_key_value_groups
                
                for proj in [attn.k_proj, attn.v_proj]:
                    kv_weight = proj.weight.view(attn.num_key_value_heads, head_size, -1)
                    kv_weight = kv_weight[kv_mask.bool()]
                    proj.weight = nn.Parameter(kv_weight.reshape(new_kv_heads * head_size, -1))
            
            # Update o_proj
            o_weight = attn.o_proj.weight.view(-1, attn.num_attention_heads, head_size)
            o_weight = o_weight[:, mask.bool(), :]
            attn.o_proj.weight = nn.Parameter(o_weight.reshape(-1, new_num_heads * head_size))
            
            # Update attention module attributes
            attn.num_attention_heads = new_num_heads
            if hasattr(attn, 'num_key_value_heads'):
                attn.num_key_value_heads = new_kv_heads
                
        return model
    
    def __call__(self, model: PreTrainedModel) -> PreTrainedModel:
        """Main entry point for pruning."""
        head_importance = self.compute_head_importance(model)
        heads_to_prune = self.select_heads_to_prune(head_importance)
        return self.prune_heads(model, heads_to_prune)
