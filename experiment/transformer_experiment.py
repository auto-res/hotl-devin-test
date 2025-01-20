import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from loranorm import LoRANorm


def load_pretrained_model(model_name="distilgpt2"):
    """Load a pretrained model and tokenizer from HuggingFace.
    
    Args:
        model_name (str): Name of the pretrained model to load
        
    Returns:
        tuple: (model, tokenizer)
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure padding token for GPT-style models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer


def apply_loranorm_to_model(model, rank=4, lora_dropout_p=0.1, lora_alpha=1.0):
    """Apply LoRANorm to specific linear layers in the transformer model.
    
    Args:
        model: The transformer model
        rank (int): Rank for LoRANorm
        lora_dropout_p (float): Dropout probability
        lora_alpha (float): LoRA scaling factor
    """
    # Target specific transformer layers (attention and feed-forward)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Only apply to attention and feed-forward layers
            if any(x in name.lower() for x in ['attn', 'attention', 'mlp', 'feed_forward']):
                lora = LoRANorm.from_linear(
                    module, rank=rank,
                    lora_dropout_p=lora_dropout_p,
                    lora_alpha=lora_alpha
                )
                nn.utils.parametrize.register_parametrization(module, "weight", lora)


def create_training_data(tokenizer, batch_size=4, seq_length=32, num_batches=10):
    """Create training data for language modeling.
    
    Args:
        tokenizer: HuggingFace tokenizer
        batch_size (int): Batch size
        seq_length (int): Sequence length
        num_batches (int): Number of batches
        
    Returns:
        list: List of (input_ids, labels) tuples for causal language modeling
    """
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a world of artificial intelligence, learning is key.",
        "Deep neural networks have transformed machine learning.",
        "Language models can understand and generate text."
    ]
    
    data = []
    for _ in range(num_batches):
        # Randomly select and combine sample texts
        batch_texts = [
            " ".join([sample_texts[i % len(sample_texts)] for i in range(3)])
            for _ in range(batch_size)
        ]
        
        # Tokenize with padding and truncation
        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=seq_length,
            return_tensors="pt"
        )
        
        # For causal language modeling, labels are input shifted by 1
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        labels = input_ids.clone()
        
        # Mask out padding tokens in labels
        labels[labels == tokenizer.pad_token_id] = -100
        
        data.append((input_ids, attention_mask, labels))
    
    return data


def train_epoch(model, data, optimizer, device='cpu'):
    """Train for one epoch using causal language modeling.
    
    Args:
        model: The model to train
        data: List of (input_ids, attention_mask, labels) tuples
        optimizer: The optimizer
        device: Device to use for training
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for input_ids, attention_mask, labels in data:
        # Move tensors to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        # Forward pass with attention mask
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Backward pass and optimization
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data)


def run_experiment(
    model_name="distilgpt2",
    batch_size=4,
    seq_length=32,
    num_epochs=5,
    rank=4,
    lora_dropout_p=0.1,
    lora_alpha=1,
    learning_rate=5e-5,
    device='cpu'
):
    """Run a LoRANorm experiment with a transformer model.
    
    Args:
        model_name (str): HuggingFace model name
        batch_size (int): Batch size
        seq_length (int): Sequence length
        num_epochs (int): Number of epochs
        rank (int): Rank for LoRANorm
        lora_dropout_p (float): Dropout probability
        lora_alpha (float): LoRA scaling factor
        learning_rate (float): Learning rate
        device (str): Device to use ('cpu' or 'cuda')
    """
    # Load model and tokenizer
    model, tokenizer = load_pretrained_model(model_name)
    model = model.to(device)
    
    # Apply LoRANorm
    apply_loranorm_to_model(model, rank, lora_dropout_p, lora_alpha)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create training data
    train_data = create_training_data(tokenizer, batch_size, seq_length)
    
    # Training loop
    print(f"\nTraining with model: {model_name}")
    print(f"LoRANorm rank: {rank}, dropout: {lora_dropout_p}, alpha: {lora_alpha}\n")
    
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_data, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Run LoRANorm experiments with transformers')
    parser.add_argument('--model-name', type=str, default='distilgpt2',
                      help='HuggingFace model name (default: distilgpt2)')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size (default: 4)')
    parser.add_argument('--seq-length', type=int, default=32,
                      help='Sequence length (default: 32)')
    parser.add_argument('--num-epochs', type=int, default=5,
                      help='Number of epochs (default: 5)')
    parser.add_argument('--rank', type=int, default=4,
                      help='LoRANorm rank (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='LoRANorm dropout probability (default: 0.1)')
    parser.add_argument('--alpha', type=float, default=1.0,
                      help='LoRANorm alpha scaling (default: 1.0)')
    parser.add_argument('--lr', type=float, default=5e-5,
                      help='Learning rate (default: 5e-5)')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cpu', 'cuda'],
                      help='Device to use (default: cpu)')
    
    args = parser.parse_args()
    
    run_experiment(
        model_name=args.model_name,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_epochs=args.num_epochs,
        rank=args.rank,
        lora_dropout_p=args.dropout,
        lora_alpha=args.alpha,
        learning_rate=args.lr,
        device=args.device
    )


if __name__ == "__main__":
    main()
