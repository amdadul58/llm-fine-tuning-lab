from transformers import AutoTokenizer


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)


# Input text
text = "Smart farming uses artificial intelligence."


# Tokenize text
tokens = tokenizer.tokenize(text)

print("Original text:")
print(text)

print("\nTokens:")
print(tokens)


# Convert tokens to token IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\nToken IDs:")
print(token_ids)


# Direct encoding
encoded = tokenizer(
    text,
    return_tensors="pt"
)

print("\nEncoded output:")
print(encoded)