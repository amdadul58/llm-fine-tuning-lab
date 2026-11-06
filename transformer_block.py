import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=4, num_heads=2, ff_dim=16):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attention_output, _ = self.attention(x, x, x)

        x = self.norm1(x + attention_output)

        ff_output = self.feed_forward(x)

        x = self.norm2(x + ff_output)

        return x


# Example input
X = torch.tensor([
    [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.5],
    ]
])

model = TransformerBlock()

output = model(X)

print("Input shape:")
print(X.shape)

print("\nOutput shape:")
print(output.shape)

print("\nTransformer Block output:")
print(output)