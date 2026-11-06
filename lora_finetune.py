import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "distilgpt2"
DATA_PATH = "data/train.json"
OUTPUT_DIR = "outputs/lora-distilgpt2"

DEVICE = "cpu"

print("Device:", DEVICE)
print("Model:", MODEL_NAME)


# ============================================================
# Load Dataset
# ============================================================

with open(DATA_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

print("Dataset samples:", len(data))


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPT-2 does not have a pad token by default.
tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# Prepare Text
# ============================================================

texts = []

for item in data:
    text = (
        "### Instruction:\n"
        + item["instruction"]
        + "\n\n"
        + "### Response:\n"
        + item["response"]
        + tokenizer.eos_token
    )

    texts.append(text)


# ============================================================
# Simple Dataset
# ============================================================

class AgricultureDataset(torch.utils.data.Dataset):

    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, index):
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }

        item["labels"] = item["input_ids"].clone()

        return item


dataset = AgricultureDataset(
    texts,
    tokenizer,
    max_length=128,
)

print("Tokenized samples:", len(dataset))


# ============================================================
# Load Pretrained Model
# ============================================================

print("\nLoading pretrained model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.config.pad_token_id = tokenizer.pad_token_id


# ============================================================
# LoRA Configuration
# ============================================================

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


# ============================================================
# Apply LoRA
# ============================================================

model = get_peft_model(
    model,
    lora_config,
)

print("\nLoRA configuration applied.")

model.print_trainable_parameters()


# ============================================================
# Training Configuration
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=2,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=1,

    learning_rate=5e-4,

    logging_steps=1,

    save_strategy="no",

    report_to="none",

    fp16=False,

    dataloader_num_workers=0,

    use_cpu=True,
)


# ============================================================
# Data Collator
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# ============================================================
# Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)


# ============================================================
# Start Training
# ============================================================

print("\nStarting LoRA fine-tuning...")

trainer.train()

print("\nTraining completed.")


# ============================================================
# Save LoRA Adapter
# ============================================================

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nLoRA adapter saved to:")
print(OUTPUT_DIR)