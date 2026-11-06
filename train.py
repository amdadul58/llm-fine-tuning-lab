import torch
import torch.nn as nn
import torch.optim as optim

# Device
device = torch.device("cpu")

print("Device:", device)
print("PyTorch:", torch.__version__)


# Simple Neural Network
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


# Training data
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
], dtype=torch.float32)

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [2.0],
], dtype=torch.float32)


# Model
model = SimpleModel().to(device)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Training
epochs = 1000

for epoch in range(epochs):

    predictions = model(X)

    loss = criterion(predictions, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss.item():.6f}"
        )


# Evaluation
model.eval()

with torch.no_grad():
    predictions = model(X)

print("\nPredictions:")

for input_data, prediction in zip(X, predictions):
    print(
        f"Input: {input_data.tolist()} "
        f"-> Prediction: {prediction.item():.4f}"
    )