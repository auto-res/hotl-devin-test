import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.loranorm import LoRANorm


def create_dummy_data(batch_size, input_dim, num_batches=10):
    """Create dummy data for demonstration purposes.
    
    In practice, replace this with your actual data loading logic.
    
    Args:
        batch_size (int): Number of samples per batch
        input_dim (int): Input dimension
        num_batches (int): Number of batches to generate
        
    Returns:
        list: List of (input, target) tuples
    """
    data = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, input_dim)
        # Dummy target: sum of inputs > 0
        y = (x.sum(dim=1) > 0).float()
        data.append((x, y))
    return data


class SimpleModel(nn.Module):
    """A simple model for demonstration purposes.
    
    Can be configured as linear, MLP, or conv model based on architecture_type.
    """
    def __init__(self, input_dim, output_dim, architecture_type='linear'):
        super().__init__()
        self.architecture_type = architecture_type
        
        if architecture_type == 'linear':
            self.model = nn.Linear(input_dim, output_dim)
        elif architecture_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.ReLU(),
                nn.Linear(input_dim * 2, output_dim)
            )
        elif architecture_type == 'conv':
            # Assumes input_dim is square (e.g., 16 = 4x4)
            size = int(input_dim ** 0.5)
            self.model = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(4 * size * size, output_dim)
            )
        else:
            raise ValueError(f"Unknown architecture type: {architecture_type}")

    def forward(self, x):
        if self.architecture_type == 'conv':
            # Reshape for conv input
            size = int(x.shape[1] ** 0.5)
            x = x.view(-1, 1, size, size)
        return self.model(x)


def add_loranorm_to_model(model, rank=4, lora_dropout_p=0.1, lora_alpha=1):
    """Add LoRANorm to a model's layers.
    
    Args:
        model (nn.Module): The model to modify
        rank (int): Rank for LoRANorm
        lora_dropout_p (float): Dropout probability
        lora_alpha (float): LoRA scaling factor
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if isinstance(module, nn.Linear):
                lora = LoRANorm.from_linear(
                    module, rank=rank,
                    lora_dropout_p=lora_dropout_p,
                    lora_alpha=lora_alpha
                )
            else:  # Conv2d
                lora = LoRANorm.from_conv2d(
                    module, rank=rank,
                    lora_dropout_p=lora_dropout_p,
                    lora_alpha=lora_alpha
                )
            # Apply LoRANorm to the layer
            nn.utils.parametrize.register_parametrization(module, "weight", lora)


def train_epoch(model, data, optimizer, criterion):
    """Train for one epoch.
    
    Args:
        model (nn.Module): The model to train
        data (list): List of (input, target) tuples
        optimizer (optim.Optimizer): The optimizer
        criterion: The loss function
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for inputs, targets in data:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(data)


def run_experiment(
    input_dim=16,
    output_dim=1,
    batch_size=32,
    num_epochs=5,
    architecture_type='linear',
    rank=4,
    lora_dropout_p=0.1,
    lora_alpha=1,
    learning_rate=0.01
):
    """Run a LoRANorm experiment.
    
    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        batch_size (int): Batch size
        num_epochs (int): Number of epochs to train
        architecture_type (str): Type of architecture ('linear', 'mlp', or 'conv')
        rank (int): Rank for LoRANorm
        lora_dropout_p (float): Dropout probability for LoRANorm
        lora_alpha (float): LoRA scaling factor
        learning_rate (float): Learning rate for optimization
    """
    # Create model and add LoRANorm
    model = SimpleModel(input_dim, output_dim, architecture_type)
    add_loranorm_to_model(model, rank, lora_dropout_p, lora_alpha)
    
    # Create optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # Create dummy data (replace with your data loading logic)
    train_data = create_dummy_data(batch_size, input_dim)
    
    # Training loop
    print(f"\nTraining with architecture: {architecture_type}")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    print(f"LoRANorm rank: {rank}, dropout: {lora_dropout_p}, alpha: {lora_alpha}\n")
    
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_data, optimizer, criterion)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Run LoRANorm experiments')
    parser.add_argument('--input-dim', type=int, default=16,
                      help='Input dimension (default: 16)')
    parser.add_argument('--output-dim', type=int, default=1,
                      help='Output dimension (default: 1)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size (default: 32)')
    parser.add_argument('--num-epochs', type=int, default=5,
                      help='Number of epochs (default: 5)')
    parser.add_argument('--architecture', type=str, default='linear',
                      choices=['linear', 'mlp', 'conv'],
                      help='Model architecture (default: linear)')
    parser.add_argument('--rank', type=int, default=4,
                      help='LoRANorm rank (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='LoRANorm dropout probability (default: 0.1)')
    parser.add_argument('--alpha', type=float, default=1.0,
                      help='LoRANorm alpha scaling (default: 1.0)')
    parser.add_argument('--lr', type=float, default=0.01,
                      help='Learning rate (default: 0.01)')
    
    args = parser.parse_args()
    
    run_experiment(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        architecture_type=args.architecture,
        rank=args.rank,
        lora_dropout_p=args.dropout,
        lora_alpha=args.alpha,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()
