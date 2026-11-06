import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "distilgpt2"
ADAPTER_PATH = "outputs/lora-distilgpt2"
DATA_PATH = "data/train.json"


print("=" * 70)
print("LLM FINE-TUNING LAB - FINAL TEST")
print("=" * 70)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Load base model
print("\nLoading base model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

base_model.config.pad_token_id = tokenizer.pad_token_id


# Load LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

model.eval()

print("Model loaded successfully.")


# ---------------------------------------------------------
# Load training dataset
# ---------------------------------------------------------

with open(DATA_PATH, "r", encoding="utf-8") as file:
    dataset = json.load(file)


# Take some questions directly from training data
training_questions = [
    dataset[0]["instruction"],
    dataset[1]["instruction"],
    dataset[2]["instruction"],
]


# Questions that were NOT used during training
unseen_questions = [
    "How does soil moisture affect irrigation decisions?",
    "What should a farmer consider before applying fertilizer?",
    "How can farmers reduce water waste in a vegetable field?",
]


def generate_answer(question):

    prompt = (
        "### Instruction:\n"
        + question
        + "\n\n"
        + "### Response:\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=60,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    return answer


# ---------------------------------------------------------
# Training-data test
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("TEST 1: QUESTIONS FROM TRAINING DATA")
print("=" * 70)

for index, question in enumerate(training_questions, start=1):

    print(f"\nQuestion {index}:")
    print(question)

    answer = generate_answer(question)

    print("\nAI Answer:")
    print(answer)


# ---------------------------------------------------------
# Unseen-question test
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("TEST 2: UNSEEN AGRICULTURE QUESTIONS")
print("=" * 70)

for index, question in enumerate(unseen_questions, start=1):

    print(f"\nQuestion {index}:")
    print(question)

    answer = generate_answer(question)

    print("\nAI Answer:")
    print(answer)


print("\n")
print("=" * 70)
print("FINAL TEST COMPLETED")
print("=" * 70)