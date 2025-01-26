"""
Wide Residual Network (WRN) implementation.
Based on "Wide Residual Networks" - https://arxiv.org/abs/1605.07146
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic residual block for WRN."""
    def __init__(self, in_planes, out_planes, stride, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = dropout

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class WideResNet(nn.Module):
    """Wide Residual Network (WRN) implementation."""
    
    def __init__(self, depth=28, width_factor=10, dropout=0.0, num_classes=100):
        super().__init__()
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        self.depth = depth  # Store depth as instance variable
        n = (depth - 4) // 6
        
        # Widths for each stage
        widths = [16, 16*width_factor, 32*width_factor, 64*width_factor]
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        # Residual stages
        self.stage1 = self._make_layer(widths[0], widths[1], n, stride=1, dropout=dropout)
        self.stage2 = self._make_layer(widths[1], widths[2], n, stride=2, dropout=dropout)
        self.stage3 = self._make_layer(widths[2], widths[3], n, stride=2, dropout=dropout)
        
        # Final batch norm and classifier
        self.bn = nn.BatchNorm2d(widths[3])
        self.linear = nn.Linear(widths[3], num_classes)
        
        # Initialize weights
        self._init_weights()

    def _make_layer(self, in_planes, out_planes, num_blocks, stride, dropout):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(in_planes, out_planes, stride, dropout))
            in_planes = out_planes
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
