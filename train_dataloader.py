import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# =========================
# Dataset
# =========================

class SimpleDataset(Dataset):

    def __init__(self):

        self.X = torch.tensor([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [3.0, 1.0],
        ], dtype=torch.float32)

        self.y = torch.tensor([
            [0.0],
            [1.0],
            [1.0],
            [2.0],
            [2.0],
            [3.0],
            [3.0],
            [4.0],
        ], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# =========================
# Model
# =========================

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


# =========================
# Device
# =========================

device = torch.device("cpu")

print("Device:", device)
print("PyTorch:", torch.__version__)


# =========================
# Dataset + DataLoader
# =========================

dataset = SimpleDataset()

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

print("Total samples:", len(dataset))


# =========================
# Model
# =========================

model = SimpleModel().to(device)


# =========================
# Loss + Optimizer
# =========================

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)


# =========================
# Training
# =========================

epochs = 500

for epoch in range(epochs):

    total_loss = 0.0

    for batch_X, batch_y in dataloader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        predictions = model(batch_X)

        # Calculate loss
        loss = criterion(predictions, batch_y)

        # Clear old gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update model
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    if (epoch + 1) % 50 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {average_loss:.6f}"
        )


# =========================
# Evaluation
# =========================

model.eval()

print("\nPredictions:")

with torch.no_grad():

    for input_data, target in dataset:

        prediction = model(
            input_data.unsqueeze(0)
        )

        print(
            f"Input: {input_data.tolist()} "
            f"-> Prediction: {prediction.item():.4f} "
            f"| Target: {target.item():.1f}"
        )
