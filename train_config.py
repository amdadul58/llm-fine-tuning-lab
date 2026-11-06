import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# ==========================================
# Configuration
# ==========================================

BATCH_SIZE = 2
LEARNING_RATE = 0.01
EPOCHS = 300

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
# Dataset
# ==========================================

dataset = SimpleDataset()

print("Total samples:", len(dataset))


# ==========================================
# Train / Validation Split
# ==========================================

train_size = int(0.8 * len(dataset))
validation_size = len(dataset) - train_size

train_dataset, validation_dataset = random_split(
    dataset,
    [train_size, validation_size]
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(validation_dataset))


# ==========================================
# DataLoaders
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# Model
# ==========================================

model = SimpleModel().to(DEVICE)


# ==========================================
# Loss + Optimizer
# ==========================================

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# Training
# ==========================================

print("\nTraining started...\n")

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        predictions = model(batch_X)

        loss = criterion(predictions, batch_y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = (
        total_train_loss / len(train_loader)
    )


    # ======================================
    # Validation
    # ======================================

    model.eval()

    total_validation_loss = 0.0

    with torch.no_grad():

        for batch_X, batch_y in validation_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            predictions = model(batch_X)

            loss = criterion(
                predictions,
                batch_y
            )

            total_validation_loss += loss.item()

    average_validation_loss = (
        total_validation_loss
        / len(validation_loader)
    )


    # ======================================
    # Print Progress
    # ======================================

    if (epoch + 1) % 50 == 0:

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {average_train_loss:.6f} "
            f"Validation Loss: {average_validation_loss:.6f}"
        )


# ==========================================
# Final Evaluation
# ==========================================

print("\nTraining completed.")

model.eval()

print("\nValidation Predictions:")

with torch.no_grad():

    for input_data, target in validation_dataset:

        prediction = model(
            input_data.unsqueeze(0)
        )

        print(
            f"Input: {input_data.tolist()} "
            f"-> Prediction: {prediction.item():.4f} "
            f"| Target: {target.item():.1f}"
        )