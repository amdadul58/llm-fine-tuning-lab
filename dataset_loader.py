import torch
from torch.utils.data import Dataset, DataLoader


# Custom Dataset
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


# Create Dataset
dataset = SimpleDataset()

print("Total samples:", len(dataset))


# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


# Show batches
print("\nBatches:")

for batch_X, batch_y in dataloader:

    print("X:", batch_X)
    print("y:", batch_y)
    print("Batch size:", len(batch_X))
    print("-" * 40)