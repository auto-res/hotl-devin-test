import torch
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset


"""
This script is a sample code to train a language model.
The experimental codebase used by our research group is maintained to be compatible with this sample code.

If a new method works in this small training code, it is guaranteed to be applicable to larger experiments.

When implementing a new method in research, it is wise to follow these guidelines:
- Implement the method as an independent module.
- Use the implemented method by importing it in this train.py.
- Therefore, the new method should be implemented as an operation on the objects present in this train.py. For example, operations that replace the internal processing or variables of the LLM should be introduced as `model = new_method(model)`. Alternatively, you can change the options of the `trl` library.
- The implementation of the LLM used in the sample code exists under `reference/llm`, so refer to it.
"""


def train():
    # SFT config
    sft_config_test = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=8,
        learning_rate=2e-4,
        logging_steps=1,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_test",
        report_to="none",
        dataset_num_proc=2,
        packing=False,
        dataset_text_field="text",
        max_seq_length=2048,  
        logging_first_step=True,
    )

    # load small LM
    checkpoint = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, device_map="auto", torch_dtype=torch.bfloat16)

    # load dataset for test
    def _format_gsm8k(dataset):
        eos_token = tokenizer.eos_token
        texts = []
        for q, a in zip(dataset['question'], dataset['answer']):
            text = f"### Input:\n{q}\n### Output:\n{a}{eos_token}"
            texts.append(text)
        return {'text': texts}
    dataset = load_dataset("openai/gsm8k", "main")
    dataset = dataset["train"].map(_format_gsm8k, batched=True)
    dataset = Dataset.from_dict(dataset[:32])

    # test training
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config_test,
    )
    trainer.train()
    return model
