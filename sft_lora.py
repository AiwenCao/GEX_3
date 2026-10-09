from datasets import load_dataset

from transformers import (
    AutoTokenizer
)

from peft import (
    AutoPeftModelForCausalLM
)

from trl import (
    SFTTrainer,
    SFTConfig
)

import random
import numpy as np
import torch

SEED = 48391

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


MODEL_PATH = (
    "models/domain_adapter"
)


## Load domain-adapted model

model = (
    AutoPeftModelForCausalLM
    .from_pretrained(
        MODEL_PATH,
        is_trainable=True,
        ignore_mismatched_sizes=True
        
    )
)


tokenizer = (
    AutoTokenizer.from_pretrained(
        MODEL_PATH
    )
)


#3 Load SFT dataset

dataset = load_dataset(
    "json",
    data_files=
        "data/sft_clean.jsonl",
    split="train"
)


dataset = dataset.train_test_split(
    test_size=0.1,
    seed=SEED
)



## Convert to prompt/completion format

def format_example(example):
    prompt_text = (
        "You are a university IT support assistant.\n\n"
        f"User: {example['prompt']}\n\n"
        "Assistant:"
    )
    completion_text = " " + example["completion"]
    return {
        "text": prompt_text + completion_text,
        "prompt": prompt_text,
        "completion": completion_text
    }


dataset = dataset.map(
    format_example
)



## Training configuration


config = SFTConfig(
    output_dir=
        "models/specialized_adapter",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-4,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    max_seq_length=256,
    use_liger_kernel=False,
    report_to="none",
    seed=SEED
)



# Trainer


trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    dataset_text_field="text"
)



# Fine-tune

trainer.train()



# Save

model.save_pretrained(
    "models/specialized_adapter"
)

tokenizer.save_pretrained(
    "models/specialized_adapter"
)