"""
QLoRA Fine-tuning — Llama 3.2 1B Instruct
Dataset : cybersecurity_lora_dataset.jsonl
Method  : 4-bit NF4 quantization + LoRA adapters via PEFT.
VRAM    : ~5-6 GB
"""

import json
import os
import torch
import aquin
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from hf_auth import login_huggingface, ensure_model

MODEL_ID       = "meta-llama/Llama-3.2-1B-Instruct"
DATA_PATH      = "cybersecurity_lora_dataset.jsonl"
OUTPUT_DIR     = "./outputs/qlora"
AQUIN_RUN_NAME = "qlora-cybersecurity"

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


class AquinCallback(TrainerCallback):
    def __init__(self, run: aquin.Run):
        self.run = run

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        metrics = {}
        if "loss" in logs:
            metrics["loss"] = float(logs["loss"])
        if "learning_rate" in logs:
            metrics["learning_rate"] = float(logs["learning_rate"])
        if metrics:
            self.run.log(step=state.global_step, **metrics)

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self.run.upload_checkpoint(model, step=state.global_step)


def _aquin_config():
    return {
        "lr":                         SFT_CFG.learning_rate,
        "epochs":                     SFT_CFG.num_train_epochs,
        "rank":                       LORA_CONFIG.r,
        "lora_alpha":                 LORA_CONFIG.lora_alpha,
        "method":                     "qlora",
        "per_device_train_batch_size": SFT_CFG.per_device_train_batch_size,
        "gradient_accumulation_steps": SFT_CFG.gradient_accumulation_steps,
        "dataset":                    DATA_PATH,
    }


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

    aquin_run = None
    callbacks = []
    if os.environ.get("AQUIN_API_KEY"):
        aquin_run = aquin.Run(
            run_name      = AQUIN_RUN_NAME,
            base_model_id = MODEL_ID,
        )
        callbacks.append(AquinCallback(aquin_run))
    else:
        print("[WARN] AQUIN_API_KEY not set — skipping Aquin logging.")

    # Pass peft_config directly — SFTTrainer handles prepare_model_for_kbit_training internally
    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        args             = SFT_CFG,
        train_dataset    = dataset,
        peft_config      = LORA_CONFIG,
        callbacks        = callbacks,
    )

    try:
        trainer.train()
        trainer.model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"QLoRA adapter saved → {OUTPUT_DIR}")
        if aquin_run:
            aquin_run.finish(config=_aquin_config())
    except Exception:
        if aquin_run:
            aquin_run.finish(config=_aquin_config(), status="failed")
        raise


if __name__ == "__main__":
    main()
