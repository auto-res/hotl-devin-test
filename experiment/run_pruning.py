import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

from src.pruning.config import PruningConfig
from src.pruning.llama import prune_llama_model


@dataclass
class ExperimentConfig:
    """Configuration for pruning experiments.
    
    Args:
        model_name: Name or path of the model to prune
        pruning_rate: Fraction of heads to prune
        dataset_name: Name of the dataset to use for evaluation
        dataset_split: Split of the dataset to use
        dataset_max_samples: Maximum number of samples to use (None for all)
        output_dir: Directory to save results
        batch_size: Batch size for training
        num_epochs: Number of epochs for training
        learning_rate: Learning rate for training
        random_seed: Random seed for reproducibility
    """
    model_name: str = "HuggingFaceTB/SmolLM-135M"
    pruning_rate: float = 0.2
    dataset_name: str = "openai/gsm8k"
    dataset_split: str = "train"
    dataset_max_samples: Optional[int] = 32
    output_dir: str = "pruning_results"
    batch_size: int = 2
    num_epochs: int = 8
    learning_rate: float = 2e-4
    random_seed: Optional[int] = 3407

def format_gsm8k(example: dict, tokenizer) -> dict:
    """Format GSM8K examples for training."""
    text = f"### Input:\n{example['question']}\n### Output:\n{example['answer']}{tokenizer.eos_token}"
    return {"text": text}

def run_pruning_experiment(config: ExperimentConfig = ExperimentConfig()) -> None:
    """Run pruning experiment with configurable parameters.
    
    Args:
        config: Experiment configuration
    """
    # Set random seed
    if config.random_seed is not None:
        torch.manual_seed(config.random_seed)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # Load and prepare dataset
    dataset = load_dataset(config.dataset_name, "main")
    dataset = dataset[config.dataset_split]
    if config.dataset_max_samples:
        dataset = dataset.select(range(config.dataset_max_samples))
    
    # Format dataset
    dataset = dataset.map(
        lambda x: format_gsm8k(x, tokenizer),
        batched=True
    )
    
    # Configure pruning
    pruning_config = PruningConfig(
        pruning_rate=config.pruning_rate,
        maintain_gqa_ratio=True,
        random_seed=config.random_seed
    )
    
    # Prune model
    print(f"Pruning model with rate {config.pruning_rate}")
    pruned_model = prune_llama_model(model, pruning_config)
    
    # Save pruned model
    pruned_model.save_pretrained(os.path.join(config.output_dir, "pruned_model"))
    
    # Configure training
    training_args = TrainingArguments(
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        logging_steps=1,
        output_dir=os.path.join(config.output_dir, "training"),
        report_to="none",
        logging_first_step=True,
    )
    
    # Train and evaluate
    trainer = SFTTrainer(
        model=pruned_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=2048,
    )
    
    print("Training pruned model")
    trainer.train()
    
    # Save final model
    trainer.save_model(os.path.join(config.output_dir, "final_model"))
    print(f"Experiment completed. Results saved to {config.output_dir}")

if __name__ == "__main__":
    # Example usage
    config = ExperimentConfig(
        pruning_rate=0.2,
        dataset_max_samples=32,  # Use small subset for testing
        output_dir="pruning_results"
    )
    run_pruning_experiment(config)
