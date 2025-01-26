"""Training script for Wide Residual Networks and their variations."""
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.wrn import WideResNet
from src.improvements import EnhancedDropoutWRN, StochasticDepthWRN, AugmentedWRN

def get_transforms(training=True):
    """Get transforms for CIFAR-100."""
    if training:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

def get_model(args):
    """Get the specified model variant."""
    if args.model == 'baseline':
        return WideResNet(depth=args.depth, width_factor=args.width_factor,
                         dropout=args.dropout, num_classes=100)
    elif args.model == 'dropout':
        return EnhancedDropoutWRN(depth=args.depth, width_factor=args.width_factor,
                                 dropout=args.dropout, num_classes=100)
    elif args.model == 'stochastic':
        return StochasticDepthWRN(depth=args.depth, width_factor=args.width_factor,
                                 death_rate=args.death_rate, num_classes=100)
    elif args.model == 'augmented':
        return AugmentedWRN(depth=args.depth, width_factor=args.width_factor,
                           num_classes=100, mixup_alpha=args.mixup_alpha,
                           cutmix_alpha=args.cutmix_alpha, mix_prob=args.mix_prob)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if isinstance(model, AugmentedWRN) and model.training:
            outputs, targets_a, targets_b, lam = model(inputs, targets)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if isinstance(model, AugmentedWRN) and model.training:
            correct += (lam * predicted.eq(targets_a).sum().float()
                       + (1 - lam) * predicted.eq(targets_b).sum().float())
        else:
            correct += predicted.eq(targets).sum().item()
            
        pbar.set_postfix({'loss': running_loss/(batch_idx+1),
                         'acc': 100.*correct/total})
    
    return running_loss/len(train_loader), 100.*correct/total

def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss/len(test_loader), 100.*correct/total

def main():
    parser = argparse.ArgumentParser(description='Train WRN models on CIFAR-100')
    # Model parameters
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline', 'dropout', 'stochastic', 'augmented'])
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--width-factor', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--death-rate', type=float, default=0.5)
    parser.add_argument('--mixup-alpha', type=float, default=1.0)
    parser.add_argument('--cutmix-alpha', type=float, default=1.0)
    parser.add_argument('--mix-prob', type=float, default=0.5)
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--subset', type=int, default=None,
                        help='Use subset of training data for testing')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loading
    print('==> Preparing data..')
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=get_transforms(True))
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=get_transforms(False))
    
    if args.subset is not None:
        trainset = Subset(trainset, range(args.subset))
        testset = Subset(testset, range(args.subset))
    
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                           shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                          shuffle=False, num_workers=2)
    
    # Create model
    print('==> Building model..')
    model = get_model(args).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, trainloader, criterion,
                                          optimizer, device, epoch)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        scheduler.step()
        
        print(f'\nEpoch {epoch}:')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}%')
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            print(f'New best accuracy: {best_acc:.3f}%')
    
    print(f'\nFinal best accuracy: {best_acc:.3f}%')

if __name__ == '__main__':
    main()
