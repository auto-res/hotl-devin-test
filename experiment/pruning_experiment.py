import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.attention_head_pruning import prune_transformer

def count_attention_heads(model):
    """Count the total number of attention heads in the model."""
    total_heads = 0
    num_layers = 0
    
    for name, module in model.named_modules():
        if any(attention_name in name.lower() for attention_name in ["attention", "attn"]):
            if hasattr(module, "num_attention_heads"):
                total_heads += module.num_attention_heads
                num_layers += 1
            elif hasattr(model.config, "num_attention_heads"):
                total_heads += model.config.num_attention_heads
                num_layers += 1
    
    return total_heads, num_layers

def main():
    parser = argparse.ArgumentParser(description="Experiment with attention head pruning")
    parser.add_argument(
        "--checkpoint",
        default="HuggingFaceTB/SmolLM-135M",
        help="HuggingFace model checkpoint to use"
    )
    parser.add_argument(
        "--prune_ratio",
        type=float,
        default=0.2,
        help="Fraction of attention heads to prune (between 0 and 1)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    args = parser.parse_args()

    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        device_map=args.device,
        torch_dtype=torch.bfloat16
    )
    
    # Count heads before pruning
    total_heads_before, num_layers = count_attention_heads(model)
    print(f"\nBefore pruning:")
    print(f"Total attention heads: {total_heads_before}")
    print(f"Number of layers: {num_layers}")
    print(f"Heads per layer: {total_heads_before // num_layers}")
    
    # Prune the model
    print(f"\nPruning {args.prune_ratio * 100:.1f}% of attention heads...")
    pruned_model = prune_transformer(model, prune_ratio=args.prune_ratio)
    
    # Count heads after pruning
    total_heads_after, _ = count_attention_heads(pruned_model)
    heads_removed = total_heads_before - total_heads_after
    
    print(f"\nAfter pruning:")
    print(f"Remaining attention heads: {total_heads_after}")
    print(f"Removed attention heads: {heads_removed}")
    print(f"Pruning ratio achieved: {heads_removed / total_heads_before:.1%}")
    
    print("\nPruning completed successfully!")
    return pruned_model

if __name__ == "__main__":
    main()
