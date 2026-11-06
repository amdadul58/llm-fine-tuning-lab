import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# Configuration
# ============================================================

BASE_MODEL = "distilgpt2"
ADAPTER_PATH = "outputs/lora-distilgpt2"


# ============================================================
# Load Tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# Load Fine-Tuned Model
# ============================================================

print("Loading agriculture fine-tuned model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL
)

base_model.config.pad_token_id = tokenizer.pad_token_id

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()


print("\n" + "=" * 60)
print("AGRIPULSE AI - FINE-TUNED AGRICULTURE LLM")
print("=" * 60)

print("\nModel loaded successfully.")
print("Ask an agriculture question.")
print("Type 'exit' to stop.\n")


# ============================================================
# Generate Answer
# ============================================================

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
            max_new_tokens=50,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Only take newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    return answer


# ============================================================
# Interactive Chat
# ============================================================

while True:

    question = input("You: ").strip()

    if question.lower() == "exit":
        print("\nExiting Agriculture AI...")
        break

    if not question:
        print("Please enter a question.\n")
        continue

    print("\nAI: Generating...\n")

    answer = generate_answer(question)

    if answer:
        print(answer)
    else:
        print("[No response generated]")

    print("\n" + "-" * 60 + "\n")


print("Agriculture AI test completed.")