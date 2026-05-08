"""
QLoRA Fine-tuning — Llama 3.2 1B Instruct
Dataset : cybersecurity_lora_dataset.jsonl
Method  : 4-bit NF4 quantization + LoRA adapters via PEFT.
VRAM    : ~5-6 GB
"""

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from hf_auth import login_huggingface, ensure_model

MODEL_ID   = "meta-llama/Llama-3.2-1B-Instruct"
DATA_PATH  = "cybersecurity_lora_dataset.jsonl"
OUTPUT_DIR = "./outputs/qlora"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.float16,
    bnb_4bit_use_double_quant = True,
)

LORA_CONFIG = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = 64,
    lora_alpha     = 128,
    lora_dropout   = 0.05,
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    bias           = "none",
)

SFT_CFG = SFTConfig(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = 3,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,
    learning_rate               = 2e-4,
    lr_scheduler_type           = "cosine",
    warmup_steps                = 10,
    logging_steps               = 10,
    save_strategy               = "epoch",
    report_to                   = "none",
    optim                       = "paged_adamw_8bit",
    fp16                        = True,
    # SFT-specific
    max_length                  = 512,
    dataset_text_field          = "text",
    packing                     = True,
)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def format_text(sample, tokenizer):
    messages = [{"role": "user",      "content": sample["user"]},
                {"role": "assistant", "content": sample["assistant"]}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    login_huggingface()
    ensure_model(MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(load_jsonl(DATA_PATH)).map(
        lambda x: {"text": format_text(x, tokenizer)},
        remove_columns=["user", "assistant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config = BNB_CONFIG,
        device_map          = "auto",
    )

    # Pass peft_config directly — SFTTrainer handles prepare_model_for_kbit_training internally
    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        args             = SFT_CFG,
        train_dataset    = dataset,
        peft_config      = LORA_CONFIG,
    )
    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"QLoRA adapter saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
