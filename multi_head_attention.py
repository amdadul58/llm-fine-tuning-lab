import torch
import torch.nn as nn

# Example token representations
X = torch.tensor([
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 0.5],
])

# Multi-Head Attention
attention = nn.MultiheadAttention(
    embed_dim=4,
    num_heads=2,
    batch_first=True
)

# Add batch dimension
X = X.unsqueeze(0)

output, weights = attention(X, X, X)

print("Input shape:")
print(X.shape)

print("\nOutput shape:")
print(output.shape)

print("\nAttention weights shape:")
print(weights.shape)

print("\nAttention output:")
print(output)

print("\nAttention weights:")
print(weights)