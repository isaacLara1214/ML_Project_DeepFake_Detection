import os
import torch
from torch.utils.data import Dataset, random_split
import cv2

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        # walk real folder
        for root, dirs, files in os.walk(os.path.join(root_dir, "real")):
            for f in files:
                if f.endswith((".jpg", ".png")):
                    self.samples.append((os.path.join(root, f), 0))

        # walk fake folder
        for root, dirs, files in os.walk(os.path.join(root_dir, "fake")):
            for f in files:
                if f.endswith((".jpg", ".png")):
                    self.samples.append((os.path.join(root, f), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, label

# build the dataset
dataset = DeepfakeDataset("data/faces")
print(f"Total images found: {len(dataset)}")

# split 70% train, 15% val, 15% test
train_size = int(0.70 * len(dataset))
val_size   = int(0.15 * len(dataset))
test_size  = len(dataset) - train_size - val_size

print(f"Train: {train_size} | Val: {val_size} | Test: {test_size}")

train_set, val_set, test_set = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # fixed seed = reproducible
)

# save the indices
torch.save({
    'train': train_set.indices,
    'val':   val_set.indices,
    'test':  test_set.indices
}, 'data/split_indices.pt')

print("Saved to data/split_indices.pt")