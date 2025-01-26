"""Enhanced Dropout variation of Wide Residual Network.
This implementation uses a higher dropout rate and applies dropout at multiple locations
in the network to improve regularization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..wrn import WideResNet, BasicBlock

class EnhancedDropoutBlock(BasicBlock):
    """BasicBlock with enhanced dropout strategy."""
    def __init__(self, in_planes, out_planes, stride, dropout=0.3):
        super().__init__(in_planes, out_planes, stride, dropout)
        self.use_dropout2 = True  # Additional dropout after shortcut

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        if self.dropout > 0:  # First dropout after ReLU
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = F.relu(out)
        if self.dropout > 0:  # Second dropout after ReLU
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)
        
        out += self.shortcut(x)
        if self.use_dropout2 and self.dropout > 0:  # Additional dropout after shortcut
            out = F.dropout(out, p=self.dropout/2, training=self.training)
        return out

class EnhancedDropoutWRN(WideResNet):
    """WRN with enhanced dropout strategy.
    
    This variation:
    1. Uses a higher default dropout rate (0.3)
    2. Applies dropout at multiple locations in each block
    3. Uses a reduced dropout rate after shortcut connections
    """
    
    def __init__(self, depth=28, width_factor=10, dropout=0.3, num_classes=100):
        # Initialize with higher dropout rate
        super().__init__(depth, width_factor, dropout, num_classes)
    
    def _make_layer(self, in_planes, out_planes, num_blocks, stride, dropout):
        """Override to use EnhancedDropoutBlock instead of BasicBlock."""
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(EnhancedDropoutBlock(in_planes, out_planes, stride, dropout))
            in_planes = out_planes
        return nn.Sequential(*layers)
