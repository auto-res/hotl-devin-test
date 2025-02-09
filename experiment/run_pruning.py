from typing import Optional, Union
from dataclasses import dataclass
import torch
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from src.pruning.head_pruning import HeadPruner

@dataclass
class PruningConfig:
    model_name: str = "HuggingFaceTB/SmolLM-135M"
    prune_ratio: float = 0.2
    importance_metric: str = "weight_magnitude"
    dataset_name: Optional[str] = "openai/gsm8k"
    dataset_split: Optional[str] = "train"
    dataset_subset: Optional[int] = None

def run_pruning_experiment(config: PruningConfig):
    """Run pruning experiment with given configuration."""
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # Initialize pruner
    pruner = HeadPruner(
        prune_ratio=config.prune_ratio,
        importance_metric=config.importance_metric
    )
    
    # Prune model
    pruned_model = pruner(model)
    
    return pruned_model

if __name__ == "__main__":
    config = PruningConfig()
    pruned_model = run_pruning_experiment(config)
