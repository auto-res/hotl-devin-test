"""Stochastic Depth variation of Wide Residual Network.
Implementation based on "Deep Networks with Stochastic Depth".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..wrn import WideResNet, BasicBlock

class StochasticDepthBlock(BasicBlock):
    """BasicBlock with stochastic depth."""
    def __init__(self, in_planes, out_planes, stride, death_rate=0.0):
        super().__init__(in_planes, out_planes, stride, dropout=0.0)
        self.death_rate = death_rate
        
    def forward(self, x):
        if not self.training or torch.rand(1) > self.death_rate:
            out = self.bn1(x)
            out = F.relu(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = F.relu(out)
            out = self.conv2(out)
            
            # Scale output during training
            if self.training:
                out = out / (1 - self.death_rate)
            
            out += self.shortcut(x)
            return out
        else:
            return self.shortcut(x)

class StochasticDepthWRN(WideResNet):
    """WRN with stochastic depth.
    
    This variation:
    1. Randomly drops entire residual blocks during training
    2. Uses linear decay of death rate from input to output
    3. Scales the output during training to maintain expected value
    """
    
    def __init__(self, depth=28, width_factor=10, death_rate=0.5, num_classes=100):
        self.depth = depth  # Store depth as instance variable
        super().__init__(depth, width_factor, dropout=0.0, num_classes=num_classes)
        self.death_rate = death_rate
        
    def _make_layer(self, in_planes, out_planes, num_blocks, stride, dropout):
        """Override to use StochasticDepthBlock with linearly increasing death rate."""
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        total_blocks = (self.depth - 4) // 2  # Total number of residual blocks
        start_block = len(layers)
        
        for i, stride in enumerate(strides):
            # Linear decay of death rate from input to output
            death_rate = self.death_rate * (start_block + i) / total_blocks
            layers.append(StochasticDepthBlock(in_planes, out_planes, stride, death_rate))
            in_planes = out_planes
            
        return nn.Sequential(*layers)
