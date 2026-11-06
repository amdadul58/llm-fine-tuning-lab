import torch
from transformers import AutoTokenizer, AutoModel


# ==========================================
# Model name
# ==========================================

MODEL_NAME = "distilbert-base-uncased"


# ==========================================
# Load Tokenizer
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ==========================================
# Load Pretrained Transformer
# ==========================================

model = AutoModel.from_pretrained(
    MODEL_NAME
)


# ==========================================
# Input text
# ==========================================

text = "Smart farming uses artificial intelligence."


# ==========================================
# Tokenize
# ==========================================

inputs = tokenizer(
    text,
    return_tensors="pt"
)


# ==========================================
# Model Inference
# ==========================================

model.eval()

with torch.no_grad():

    outputs = model(**inputs)


# ==========================================
# Display Information
# ==========================================

print("Model:", MODEL_NAME)

print("\nInput text:")
print(text)

print("\nInput IDs shape:")
print(inputs["input_ids"].shape)

print("\nModel output shape:")
print(outputs.last_hidden_state.shape)