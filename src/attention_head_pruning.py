import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional
from transformers import PreTrainedModel

def compute_head_importance(model: PreTrainedModel) -> torch.Tensor:
    """
    Compute importance scores for each attention head based on weight magnitudes.
    
    Args:
        model: A HuggingFace transformer model
        
    Returns:
        torch.Tensor: Importance scores with shape [num_layers, num_heads]
    """
    head_importance = []
    
    # Iterate through transformer layers
    for name, module in model.named_modules():
        # Look for attention modules
        if any(attention_name in name.lower() for attention_name in ["attention", "attn"]):
            # Get query weights which we'll use as proxy for importance
            if hasattr(module, "q_proj"):  # GPT style
                q_weight = module.q_proj.weight
            elif hasattr(module, "query"):  # BERT style
                q_weight = module.query.weight
            else:
                continue
                
            # Get model config to determine number of attention heads
            config = getattr(model, "config", None)
            if config is None:
                continue
                
            num_attention_heads = getattr(config, "num_attention_heads", None)
            if num_attention_heads is None:
                continue
                
            hidden_size = q_weight.shape[0]
            head_size = hidden_size // num_attention_heads
            
            # Reshape weights to separate heads
            q_weight = q_weight.view(num_attention_heads, head_size, -1)
            
            # Compute importance as average magnitude of weights
            head_scores = q_weight.abs().mean(dim=(1, 2))
            head_importance.append(head_scores.detach().cpu())
    
    if not head_importance:
        raise ValueError("Could not find any attention heads in the model")
        
    # Stack importance scores from all layers
    head_importance = torch.stack(head_importance)
    return head_importance

def select_heads_to_prune(
    importance_scores: torch.Tensor,
    prune_ratio: float = 0.2
) -> Dict[int, List[int]]:
    """
    Select which heads to prune based on importance scores.
    
    Args:
        importance_scores: Tensor of shape [num_layers, num_heads] with importance scores
        prune_ratio: Fraction of heads to prune (between 0 and 1)
        
    Returns:
        Dict mapping layer indices to lists of head indices to prune
    """
    num_layers, num_heads = importance_scores.shape
    total_heads = num_layers * num_heads
    num_heads_to_prune = int(prune_ratio * total_heads)
    
    # Flatten scores and get indices of least important heads
    flattened_scores = importance_scores.view(-1)
    _, indices_to_prune = torch.topk(flattened_scores,
                                   k=num_heads_to_prune,
                                   largest=False)
    
    # Convert flat indices to layer and head indices
    layers_to_prune = {}
    for idx in indices_to_prune:
        layer_idx = idx.item() // num_heads
        head_idx = idx.item() % num_heads
        if layer_idx not in layers_to_prune:
            layers_to_prune[layer_idx] = []
        layers_to_prune[layer_idx].append(head_idx)
    
    return layers_to_prune

def prune_attention_heads(
    attention_module: nn.Module,
    head_mask: torch.Tensor,
    config: PreTrainedModel
) -> None:
    """
    Prune specific attention heads from an attention module.
    
    Args:
        attention_module: The attention module to prune
        head_mask: Boolean tensor indicating which heads to keep
        config: Model configuration
    """
    # Helper function to prune linear layers
    def prune_linear_layer(layer: nn.Linear,
                          mask: torch.Tensor,
                          dim: int = 0,
                          preserve_output_size: bool = False,
                          head_size: Optional[int] = None) -> nn.Linear:
        """Prune a linear layer to keep only certain indices along a dimension.
        
        Args:
            layer: Linear layer to prune
            mask: Boolean mask indicating which indices to keep
            dim: Dimension to prune (0 for output features, 1 for input features)
            preserve_output_size: If True, maintain the original output size
            head_size: Size of each attention head
        """
        # Move mask to the same device as the layer
        mask = mask.to(layer.weight.device)
        
        # Get the indices to keep
        indices = torch.nonzero(mask).squeeze()
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
            
        # Ensure we're not trying to prune everything
        if len(indices) == 0:
            print("Warning: Cannot prune all values. Keeping the first one.")
            indices = torch.tensor([0], device=layer.weight.device)
            
        # Ensure indices are within bounds
        max_dim = layer.weight.size(dim)
        valid_indices = indices[indices < max_dim]
        
        if dim == 1 and preserve_output_size:
            # For output projection, we need to handle the concatenated q/kv heads
            if head_size is None:
                raise ValueError("head_size must be provided when preserve_output_size=True")
                
            # Calculate dimensions for the new layer
            head_mask = mask.view(-1)[:num_heads]
            remaining_heads = head_mask.sum().item()
            new_input_size = remaining_heads * head_size
            
            # Reshape weight matrix to work with heads
            old_weight = layer.weight
            num_heads = old_weight.size(1) // head_size
            
            # Reshape to [hidden_size, num_heads, head_size]
            W = old_weight.view(old_weight.size(0), num_heads, head_size)
            
            # Create head-level mask by reshaping the input mask
            head_mask = mask.view(-1)[:num_heads]
            
            # Select only the heads we want to keep
            W = W[:, head_mask.bool(), :]
            
            # Reshape back to [hidden_size, new_input_size]
            W = W.reshape(old_weight.size(0), new_input_size)
            
            # Scale the weights to maintain output magnitude
            W = W * (num_heads / remaining_heads) ** 0.5
            
            # Create new layer with same output size but adjusted input size
            new_layer = nn.Linear(new_input_size, layer.out_features,
                                bias=layer.bias is not None).to(layer.weight.device)
            
            # Set the weights and bias
            new_layer.weight.data = W
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.clone()
        else:
            # For query/key/value projections (dim=0), standard pruning
            W = layer.weight.index_select(dim, valid_indices).clone()
            if layer.bias is not None:
                b = layer.bias[valid_indices].clone() if dim == 0 else layer.bias.clone()
            else:
                b = None
            
            # Create new layer with appropriate dimensions
            if dim == 0:
                new_layer = nn.Linear(layer.in_features, len(valid_indices),
                                    bias=layer.bias is not None).to(layer.weight.device)
            else:
                new_layer = nn.Linear(len(valid_indices), layer.out_features,
                                    bias=layer.bias is not None).to(layer.weight.device)
            
            # Set the weights and bias
            new_layer.weight.data = W
            if b is not None:
                new_layer.bias.data = b
        
        # Initialize the new layer with the pruned weights
        new_layer.weight.data = W
        if b is not None:
            new_layer.bias.data = b
            
        return new_layer

    # Get the attention configuration
    head_size = attention_module.head_dim if hasattr(attention_module, 'head_dim') else (
        config.hidden_size // config.num_attention_heads)
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    hidden_size = config.hidden_size
    
    # For grouped-query attention, we need to map query head indices to key-value head indices
    kv_head_mapping = torch.arange(num_heads) * num_kv_heads // num_heads
    kv_head_mask = torch.zeros(num_kv_heads, dtype=torch.bool, device=head_mask.device)
    for i, keep_head in enumerate(head_mask):
        if keep_head:
            kv_head_mask[kv_head_mapping[i]] = True
    
    # Ensure at least one head is kept
    if head_mask.sum().item() == 0:
        print("Warning: Cannot prune all heads. Keeping at least one.")
        head_mask[0] = True
        kv_head_mask[0] = True
    
    # Create masks for query and key-value projections
    q_mask = torch.zeros(hidden_size, dtype=torch.bool, device=head_mask.device)
    kv_mask = torch.zeros(hidden_size, dtype=torch.bool, device=head_mask.device)
    
    # Set masks for query heads
    q_indices = torch.nonzero(head_mask).squeeze()
    if q_indices.dim() == 0:
        q_indices = q_indices.unsqueeze(0)
    for idx in q_indices:
        start_idx = idx.item() * head_size
        end_idx = (idx.item() + 1) * head_size
        q_mask[start_idx:end_idx] = True
    
    # Set masks for key-value heads
    kv_indices = torch.nonzero(kv_head_mask).squeeze()
    if kv_indices.dim() == 0:
        kv_indices = kv_indices.unsqueeze(0)
    for idx in kv_indices:
        start_idx = idx.item() * head_size
        end_idx = (idx.item() + 1) * head_size
        kv_mask[start_idx:end_idx] = True
    
    # For output projection in grouped-query attention:
    # 1. Query heads project from hidden_size to (num_heads * head_dim)
    # 2. Key-value heads project from hidden_size to (num_kv_heads * head_dim)
    qkv_mask = torch.zeros(hidden_size, dtype=torch.bool, device=head_mask.device)
    
    # Calculate the total size needed for remaining heads
    num_remaining_q_heads = len(q_indices)
    num_remaining_kv_heads = len(kv_indices)
    total_head_size = (num_remaining_q_heads + num_remaining_kv_heads) * head_size
    
    # Create mask for concatenated q and kv features
    current_idx = 0
    
    # Add query heads to the mask
    for idx in q_indices:
        if current_idx + head_size <= hidden_size:
            qkv_mask[current_idx:current_idx + head_size] = True
            current_idx += head_size
            
    # Add key-value heads to the mask
    for idx in kv_indices:
        if current_idx + head_size <= hidden_size:
            qkv_mask[current_idx:current_idx + head_size] = True
            current_idx += head_size
    
    # For LLaMA/GPT style attention
    if hasattr(attention_module, 'q_proj'):
        # Prune query projection using query mask
        attention_module.q_proj = prune_linear_layer(attention_module.q_proj, q_mask, dim=0)
        # Prune key and value projections using key-value mask
        attention_module.k_proj = prune_linear_layer(attention_module.k_proj, kv_mask, dim=0)
        attention_module.v_proj = prune_linear_layer(attention_module.v_proj, kv_mask, dim=0)
        
        # For output projection, we need to handle both q and kv dimensions
        if hasattr(attention_module, 'o_proj'):  # LLaMA style
            attention_module.o_proj = prune_linear_layer(attention_module.o_proj, qkv_mask, dim=1, preserve_output_size=True, head_size=head_size)
        else:  # GPT style
            attention_module.out_proj = prune_linear_layer(attention_module.out_proj, qkv_mask, dim=1, preserve_output_size=True)
    elif hasattr(attention_module, 'query'):  # BERT style
        attention_module.query = prune_linear_layer(attention_module.query, q_mask, dim=0)
        attention_module.key = prune_linear_layer(attention_module.key, kv_mask, dim=0)
        attention_module.value = prune_linear_layer(attention_module.value, kv_mask, dim=0)
        attention_module.output.dense = prune_linear_layer(attention_module.output.dense,
                                                         qkv_mask,
                                                         dim=1,
                                                         preserve_output_size=True,
                                                         head_size=head_size)

def prune_transformer(
    model: PreTrainedModel,
    prune_ratio: float = 0.2
) -> PreTrainedModel:
    """
    Prune attention heads from a transformer model.
    
    Args:
        model: HuggingFace transformer model to prune
        prune_ratio: Fraction of heads to prune (between 0 and 1)
        
    Returns:
        Pruned model
    """
    # Compute importance scores for all heads
    importance_scores = compute_head_importance(model)
    
    # Select heads to prune
    layers_to_prune = select_heads_to_prune(importance_scores, prune_ratio)
    
    # Create a copy of the model to avoid modifying the original
    model = model.cpu()
    pruned_model = type(model)(model.config)
    pruned_model.load_state_dict(model.state_dict())
    
    # Prune each layer
    for name, module in pruned_model.named_modules():
        if any(attention_name in name.lower() for attention_name in ["attention", "attn"]):
            layer_idx = int(''.join(filter(str.isdigit, name)))
            if layer_idx in layers_to_prune:
                # Create head mask (1 for heads to keep, 0 for heads to prune)
                head_mask = torch.ones(model.config.num_attention_heads, dtype=torch.bool)
                head_mask[layers_to_prune[layer_idx]] = 0
                
                # Prune the attention module
                prune_attention_heads(module, head_mask, model.config)
    
    # Update the config to reflect the new number of attention heads
    total_heads = model.config.num_attention_heads
    pruned_heads = sum(len(heads) for heads in layers_to_prune.values())
    model.config.num_attention_heads = total_heads - pruned_heads
    
    return pruned_model.to(model.device)
