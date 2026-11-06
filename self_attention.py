import torch
import torch.nn.functional as F


# ==========================================
# Simple token representations
# ==========================================

tokens = [
    "I",
    "love",
    "smart",
    "farming"
]


# Each token has a small 4-dimensional vector
X = torch.tensor([
    [1.0, 0.0, 0.0, 1.0],  # I
    [0.0, 1.0, 0.0, 1.0],  # love
    [0.0, 0.0, 1.0, 1.0],  # smart
    [0.0, 0.0, 1.0, 0.5],  # farming
])


print("Tokens:")
print(tokens)

print("\nInput representation shape:")
print(X.shape)


# ==========================================
# Query, Key, Value
# ==========================================

W_Q = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

W_K = W_Q.clone()

W_V = W_Q.clone()


Q = X @ W_Q
K = X @ W_K
V = X @ W_V


# ==========================================
# Attention Scores
# ==========================================

scores = Q @ K.T

print("\nAttention scores:")
print(scores)


# ==========================================
# Scale scores
# ==========================================

d_k = K.shape[-1]

scaled_scores = scores / (d_k ** 0.5)


# ==========================================
# Softmax
# ==========================================

attention_weights = F.softmax(
    scaled_scores,
    dim=-1
)


print("\nAttention weights:")
print(attention_weights)


# ==========================================
# Attention Output
# ==========================================

attention_output = attention_weights @ V


print("\nAttention output:")
print(attention_output)


# ==========================================
# Show attention relationships
# ==========================================

print("\nAttention relationships:")

for i, token in enumerate(tokens):

    print(f"\n'{token}' attends to:")

    for j, other_token in enumerate(tokens):

        weight = attention_weights[i][j].item()

        print(
            f"  {other_token}: {weight:.4f}"
        )