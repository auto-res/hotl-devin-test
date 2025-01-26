"""Advanced augmentation strategies for Wide Residual Network.
Implements CutMix and Mixup augmentation strategies.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..wrn import WideResNet

def mixup_data(x, y, alpha=1.0):
    """Performs mixup on the input data and label."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """Performs cutmix on the input data and label."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    # Generate random box
    W, H = x.size()[2:]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply cutmix
    x_mixed = x.clone()
    x_mixed[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return x_mixed, y_a, y_b, lam

class AugmentedWRN(WideResNet):
    """WRN with advanced data augmentation strategies.
    
    This variation:
    1. Implements both Mixup and CutMix augmentation
    2. Randomly applies either strategy during training
    3. Maintains original architecture but modifies the training process
    """
    
    def __init__(self, depth=28, width_factor=10, num_classes=100, 
                 mixup_alpha=1.0, cutmix_alpha=1.0, mix_prob=0.5):
        super().__init__(depth, width_factor, dropout=0.0, num_classes=num_classes)
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        
    def forward_train(self, x, y):
        """Forward pass during training with augmentation."""
        if torch.rand(1) < self.mix_prob:
            if torch.rand(1) < 0.5:  # Randomly choose between Mixup and CutMix
                x_mixed, y_a, y_b, lam = mixup_data(x, y, self.mixup_alpha)
            else:
                x_mixed, y_a, y_b, lam = cutmix_data(x, y, self.cutmix_alpha)
            
            output = super().forward(x_mixed)
            return output, y_a, y_b, lam
        
        output = super().forward(x)
        return output, y, y, 1.0
        
    def forward(self, x, y=None):
        """Forward pass with optional augmentation during training."""
        if self.training and y is not None:
            return self.forward_train(x, y)
        return super().forward(x)
