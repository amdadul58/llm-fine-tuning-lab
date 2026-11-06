import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

text = "Smart farming uses artificial intelligence to"

inputs = tokenizer(text, return_tensors="pt")

model.eval()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False
    )

generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nInput:")
print(text)

print("\nGenerated text:")
print(generated_text)