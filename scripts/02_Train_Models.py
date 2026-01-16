import os
import json
import random
from PIL import Image

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

# ## Configuration 


# CONFIG

image_size = 224
batch_size = 8          # small for CPU
num_workers = 0        # safer on Windows / local
seed = 42
MAX_IMAGES = 4000      
EPOCHS = 5             

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device="cpu"

# ## Setting Paths


# PATHS

cwd = os.getcwd()

project_root = os.path.abspath(os.path.join(cwd, ".."))

data_root = os.path.join(project_root, "data", "raw", "AerialWaste")

image_dirs = [os.path.join(data_root, f"images{i}") for i in range(6)]

train_json = f'{data_root}/training.json'

test_json = f'{data_root}/testing.json'

# HELPERS
def get_image_path(file_name, image_dirs):
    for dir_path in image_dirs:
        full_path = os.path.join(dir_path, file_name)
        if os.path.exists(full_path):
            return full_path
    return None

# ## Image Path Validation, shuffling and limits


# LOAD METADATA

with open(train_json, "r") as f:
    train_json_data = json.load(f)

records = []
for img in train_json_data["images"]:
    path = get_image_path(img["file_name"], image_dirs)
    if path is not None:
        records.append({
            "file_name": img["file_name"],
            "full_path": path,
            "waste": int(img["is_candidate_location"])
        })

df = pd.DataFrame(records)

# SHUFFLE + LIMIT TO 4K
df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
df = df.iloc[:MAX_IMAGES]

print("Total images used:", len(df))
print(df["waste"].value_counts())


# ## Train and Val split


# TRAIN / VAL SPLIT

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["waste"],
    random_state=seed,
    shuffle=True
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))

# ## Transformation Functions with Data Augmentation


# TRANSFORMS

train_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ## WasteDataset Definition


# DATASET

class WasteDataset(Dataset):
    def __init__(self, dataframe, transforms=None):
        self.df = dataframe.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["full_path"]).convert("RGB")
        label = row["waste"]

        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(label, dtype=torch.long)

train_dataset = WasteDataset(train_df, train_transforms)
val_dataset = WasteDataset(val_df, val_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

# ## Model 1 - Using RESNET-18 


# MODEL

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)


# TRAIN / EVAL

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    preds, targets = [], []

    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        preds.extend(predictions.cpu().numpy())
        targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(targets, preds)

    return avg_loss, acc


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    preds, targets = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            preds.extend(predictions.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(targets, preds)

    return avg_loss, acc


# TRAIN LOOP

def get_train_val_metrics():
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
    
        val_loss, val_acc = evaluate(
            model, val_loader, criterion
        )
    
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print("-" * 40)


# ## Parameter tuning & Optimization for RESNET-18


for lr in [0.001, 0.0005, 0.01]:
    print(f"\nTraining with LR = {lr}")

    model = models.resnet18(weights="IMAGENET1K_V1")
    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    get_train_val_metrics()


lr = 0.001
EPOCHS = 5

for dr in [0.1, 0.2, 0.3, 0.5]:
    print(f"Training with Dropout = {dr}")

    # Model
    model = models.resnet18(weights="IMAGENET1K_V1")

    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    # Classifier with current dropout
    model.fc = nn.Sequential(
        nn.Dropout(p=dr),
        nn.Linear(model.fc.in_features, 2)
    )

    model = model.to(device)

    # Optimizer & loss
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train + validate
    get_train_val_metrics()


# ## Tuned parameters for RESNET-18
# 
# - Learning Rate - 0.001
# - Epoch - 5
# - Dropout Rate - 0.2


# ## Model 2 - Using RESNET-50


lr = 0.001
EPOCHS = 5

for dr in [0.1, 0.2, 0.3, 0.5]:
    print(f"Training with Dropout = {dr}")

    # Load pretrained ResNet50
    model = models.resnet50(weights="IMAGENET1K_V1")

    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    # Replace classifier
    model.fc = nn.Sequential(
        nn.Dropout(p=dr),
        nn.Linear(model.fc.in_features, 2)
    )

    model = model.to(device)

    # Optimizer & loss
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train + validate
    get_train_val_metrics()


# ## Tuned parameters for RESNET-50
# 
# - Learning Rate - 0.001
# - Epoch - 5
# - Dropout Rate - 0.2


# ## Model 3 - EFFICIENTNET-B0 


lr = 0.001
EPOCHS = 5

for dr in [0.1, 0.2, 0.3, 0.5]:
    print(f"Training with Dropout = {dr}")

    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=dr),
        nn.Linear(model.classifier[1].in_features, 2)  # original: classifier[1] is Linear
    )

    model = model.to(device)

    # Optimizer & loss
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train + validate
    get_train_val_metrics()

# ## Tuned parameters for EFFICIENTNET-B0
# 
# - Learning Rate - 0.001
# - Epoch - 5
# - Dropout Rate - 0.2

lr = 0.001
EPOCHS = 5
dr = 0.2 

# Load pretrained EfficientNet-B0
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# Freeze backbone
for p in model.parameters():
    p.requires_grad = False

# Replace classifier with best dropout
model.classifier = nn.Sequential(
    nn.Dropout(p=dr),
    nn.Linear(model.classifier[1].in_features, 2)
)

model = model.to(device)

# Optimizer & loss
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Train + validate using best hyperparameters
get_train_val_metrics()


# ## Saving the final model

model_dir = os.path.join(project_root, "model")
model_path = os.path.join(model_dir, "efficientnet_b0_best.pth")  # using .pth as convention

val_loss, val_acc = evaluate(model, val_loader, criterion)

torch.save({
    "model_state_dict": model.state_dict(),
    "dropout_rate": dr,
    "val_accuracy": val_acc
}, model_path)

print(f"Model saved as 'efficientnet_b0_best.pth' to: {model_path}")

# ## Testing the model on Test image set


# Load test JSON
with open(test_json, "r") as f:
    test_json_data = json.load(f)

# Prepare test dataframe
records = []
for img in test_json_data["images"]:
    path = get_image_path(img["file_name"], image_dirs)
    if path:
        records.append({
            "file_name": img["file_name"],
            "full_path": path,
            "waste": int(img["is_candidate_location"])
        })

test_df = pd.DataFrame(records)

# Create Dataset & DataLoader
test_dataset = WasteDataset(test_df, val_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
