from unsloth import FastModel
import torch

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    # Pretrained models
    "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",
    # Other Gemma 3 quants
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
]  # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3n-E4B-it",
    dtype=None,  # None for auto detection
    max_seq_length=1024,  # Choose any for long context!
    load_in_4bit=True,  # 4 bit quantization to reduce memory
    full_finetuning=False,  # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,  # Turn off for just text!
    finetune_language_layers=True,  # Should leave on!
    finetune_attention_modules=True,  # Attention good for GRPO
    finetune_mlp_modules=True,  # SHould leave on always!
    r=8,  # Larger = higher accuracy, but might overfit
    lora_alpha=8,  # Recommended alpha == r at least
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)

from datasets import load_dataset

dataset = load_dataset("./resources/dataset/skills.parquet", split="train[:3000]")

from unsloth.chat_templates import standardize_data_formats

dataset = standardize_data_formats(dataset)
