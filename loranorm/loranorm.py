import math
import torch
import torch.nn as nn
from torch import linalg as LA


class LoRANorm(nn.Module):
    """LoRANorm: Low-Rank Adaptation with Weight Normalization.
    
    This class implements LoRA (Low-Rank Adaptation) combined with weight normalization
    for improved fine-tuning of neural networks. It supports various layer types including
    linear, convolutional, and embedding layers.
    
    Args:
        fan_in (int): Input dimension
        fan_out (int): Output dimension
        fan_in_fan_out (bool): If True, weight is transposed. Default: False
        rank (int): Rank of the low-rank approximation. Default: 4
        lora_dropout_p (float): Dropout probability for LoRA layers. Default: 0.0
        lora_alpha (float): Scaling factor for LoRA. Default: 1
    """
    def __init__(self, fan_in, fan_out, fan_in_fan_out=False, rank=4, lora_dropout_p=0.0, lora_alpha=1.0):
        super().__init__()
        # Handle weight matrix layout
        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)
        
        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(self.swap((rank, fan_in))))
        self.lora_B = nn.Parameter(torch.zeros(self.swap((fan_out, rank))))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # Initialize using kaiming uniform
        
        # LoRA scaling (ensure float)
        self.lora_alpha = float(lora_alpha)
        self.rank = rank
        self.scaling = self.lora_alpha / float(rank)
        
        # Dropout handling
        self.lora_dropout = nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if lora_dropout_p > 0 else lambda x: x
        self.register_buffer("lora_dropout_mask", torch.ones(self.swap((1, fan_in)), dtype=self.lora_A.dtype))
        
        # Weight normalization parameters
        self.weight_g = nn.Parameter(self.norm_except_dim_0(self.lora_B))
        self.weight_v = nn.Parameter(self.lora_B.clone())
        
        # Forward function state
        self.forward_fn = self.lora_forward

    def _dropout(self, A):
        """Apply dropout to the input tensor."""
        return A * self.lora_dropout(self.lora_dropout_mask)

    def compute_weight(self):
        """Compute the normalized weight matrix."""
        g = self.weight_g
        v = self.weight_v
        return g * v / self.norm_except_dim_0(v)

    def lora_forward(self, X):
        """Forward pass with LoRA and weight normalization."""
        w_norm = self.compute_weight()
        return X + torch.matmul(*self.swap((w_norm, self.dropout_fn(self.lora_A)))).view(X.shape) * self.scaling

    def forward(self, X):
        """Forward pass through the layer."""
        return self.forward_fn(X)

    def disable_lora(self):
        """Disable LoRA adaptation."""
        self.forward_fn = lambda x: x

    def enable_lora(self):
        """Enable LoRA adaptation."""
        self.forward_fn = self.lora_forward

    @staticmethod
    def norm_except_dim_0(weight):
        """Compute L2 norm of all dimensions except dim 0."""
        output_size = (weight.size(0),) + (1,) * (weight.dim() - 1)
        out = LA.norm(weight.view(weight.size(0), -1), ord=2, dim=1).view(*output_size)
        return out

    @classmethod
    def from_linear(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1.0):
        """Create LoRANorm from a linear layer.
        
        Args:
            layer (nn.Linear): Linear layer to convert
            rank (int): Rank for low-rank approximation
            lora_dropout_p (float): Dropout probability
            lora_alpha (float): LoRA scaling factor
            
        Returns:
            LoRANorm: Initialized LoRANorm module
        """
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=False,
            rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )

    @classmethod
    def from_conv2d(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        """Create LoRANorm from a Conv2d layer."""
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=False,
            rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )

    @classmethod
    def from_embedding(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        """Create LoRANorm from an embedding layer."""
        fan_in, fan_out = layer.weight.shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=True,
            rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )
