import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ==========================================
# Configuration
# ==========================================

BATCH_SIZE = 2
EPOCHS = 100

LEARNING_RATES = [
    0.1,
    0.01,
    0.001
]

DEVICE = torch.device("cpu")


# ==========================================
# Dataset
# ==========================================

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
            [4.0, 0.0],
            [4.0, 1.0],
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
            [4.0],
            [5.0],
        ], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# ==========================================
# Model
# ==========================================

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


# ==========================================
# Dataset + DataLoader
# ==========================================

dataset = SimpleDataset()

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==========================================
# Learning Rate Experiment
# ==========================================

for learning_rate in LEARNING_RATES:

    print("\n" + "=" * 50)
    print(f"Learning Rate: {learning_rate}")
    print("=" * 50)

    # New model for each experiment
    model = SimpleModel().to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Training
    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for batch_X, batch_y in dataloader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            predictions = model(batch_X)

            loss = criterion(
                predictions,
                batch_y
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(dataloader)
        )

        if (epoch + 1) % 20 == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Loss: {average_loss:.6f}"
            )

    # Final evaluation
    model.eval()

    with torch.no_grad():

        predictions = model(dataset.X)

        final_loss = criterion(
            predictions,
            dataset.y
        )

    print(
        f"Final Loss: {final_loss.item():.6f}"
    )