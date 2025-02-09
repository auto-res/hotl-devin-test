from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention

from .config import PruningConfig
from .importance import get_importance_metric

def select_heads_to_prune(
    importance_scores: Dict[int, torch.Tensor],
    pruning_rate: float,
    maintain_gqa_ratio: bool = True,
    num_kv_heads: int = 3
) -> Dict[int, List[int]]:
    """Select which heads to prune based on importance scores.
    
    Args:
        importance_scores: Dict mapping layer indices to head importance scores
        pruning_rate: Fraction of heads to prune (between 0 and 1)
        maintain_gqa_ratio: Whether to maintain grouped-query attention ratio
        num_kv_heads: Number of key-value heads in the model
        
    Returns:
        Dict mapping layer indices to lists of head indices to prune
    """
    heads_to_prune = {}
    
    # Calculate number of heads to prune while maintaining GQA ratio
    total_heads = sum(scores.size(0) for scores in importance_scores.values())
    num_heads_to_prune = int(total_heads * pruning_rate)
    
    if maintain_gqa_ratio:
        # Ensure we prune in multiples of num_kv_heads to maintain ratio
        num_heads_to_prune = (num_heads_to_prune // num_kv_heads) * num_kv_heads
    
    # Flatten importance scores across all layers
    all_scores = []
    layer_indices = []
    head_indices = []
    for layer_idx, scores in importance_scores.items():
        all_scores.extend(scores.tolist())
        layer_indices.extend([layer_idx] * len(scores))
        head_indices.extend(range(len(scores)))
    
    # Sort by importance (ascending)
    sorted_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i])
    
    # Select heads to prune
    for idx in sorted_indices[:num_heads_to_prune]:
        layer_idx = layer_indices[idx]
        head_idx = head_indices[idx]
        
        if layer_idx not in heads_to_prune:
            heads_to_prune[layer_idx] = []
        heads_to_prune[layer_idx].append(head_idx)
    
    return heads_to_prune

def prune_attention_heads(
    attention: LlamaAttention,
    heads_to_prune: List[int],
    maintain_gqa_ratio: bool = True
) -> None:
    """Prune specified attention heads from a LlamaAttention module.
    
    Args:
        attention: The attention module to prune
        heads_to_prune: List of head indices to prune
        maintain_gqa_ratio: Whether to maintain grouped-query attention ratio
    """
    if not heads_to_prune:
        return
        
    # Sort heads to prune in descending order
    heads_to_prune = sorted(heads_to_prune, reverse=True)
    
    # Calculate new dimensions
    old_num_heads = attention.num_attention_heads
    old_num_kv_heads = attention.num_key_value_heads
    new_num_heads = old_num_heads - len(heads_to_prune)
    
    if maintain_gqa_ratio:
        # Calculate new number of KV heads
        heads_per_kv = old_num_heads // old_num_kv_heads
        new_num_kv_heads = new_num_heads // heads_per_kv
        
        # Update module attributes
        attention.num_key_value_heads = new_num_kv_heads
        attention.num_key_value_groups = new_num_heads // new_num_kv_heads
    
    attention.num_attention_heads = new_num_heads
    head_dim = attention.head_dim
    
    # Helper function to prune linear layer
    def prune_linear(layer: nn.Linear, is_output: bool = False) -> nn.Linear:
        old_weight = layer.weight.data
        old_bias = layer.bias.data if layer.bias is not None else None
        
        if is_output:
            # For output projection, we need to handle multiple heads
            old_shape = old_weight.shape
            weight = old_weight.view(old_shape[0], old_num_heads, -1)
            # Remove pruned heads
            for head in heads_to_prune:
                weight = torch.cat([weight[:, :head], weight[:, head+1:]], dim=1)
            weight = weight.reshape(old_shape[0], -1)
        else:
            # For Q/K/V projections
            weight = old_weight.view(old_num_heads, head_dim, -1)
            # Remove pruned heads
            for head in heads_to_prune:
                weight = torch.cat([weight[:head], weight[head+1:]], dim=0)
            weight = weight.reshape(-1, old_weight.shape[1])
            
            if old_bias is not None:
                bias = old_bias.view(old_num_heads, head_dim)
                for head in heads_to_prune:
                    bias = torch.cat([bias[:head], bias[head+1:]], dim=0)
                old_bias = bias.reshape(-1)
        
        # Create new layer
        new_layer = nn.Linear(
            layer.in_features,
            weight.shape[0],
            bias=layer.bias is not None
        )
        new_layer.weight.data = weight
        if old_bias is not None:
            new_layer.bias.data = old_bias
            
        return new_layer
    
    # Prune attention layers
    attention.q_proj = prune_linear(attention.q_proj)
    if maintain_gqa_ratio and heads_to_prune:
        # For GQA, we need to prune K/V heads proportionally
        kv_heads_to_prune = [h // (old_num_heads // old_num_kv_heads) for h in heads_to_prune]
        kv_heads_to_prune = sorted(set(kv_heads_to_prune), reverse=True)
        attention.k_proj = prune_linear(attention.k_proj)
        attention.v_proj = prune_linear(attention.v_proj)
    attention.o_proj = prune_linear(attention.o_proj, is_output=True)

def prune_llama_model(
    model: nn.Module,
    config: PruningConfig = PruningConfig()
) -> nn.Module:
    """Prune attention heads in LlamaForCausalLM model.
    
    Args:
        model: The model to prune
        config: Pruning configuration
        
    Returns:
        Pruned model
    """
    # Get importance metric
    importance_metric = get_importance_metric(config)
    
    # Compute importance scores
    importance_scores = importance_metric(model)
    
    # Select heads to prune
    heads_to_prune = select_heads_to_prune(
        importance_scores,
        config.pruning_rate,
        config.maintain_gqa_ratio,
        num_kv_heads=model.config.num_key_value_heads
    )
    
    # Prune heads in each layer
    for layer_idx, heads in heads_to_prune.items():
        layer = model.model.layers[layer_idx]
        prune_attention_heads(
            layer.self_attn,
            heads,
            config.maintain_gqa_ratio
        )
    
    return model
