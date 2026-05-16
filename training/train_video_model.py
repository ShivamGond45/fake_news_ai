import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn, optim
import os

# =========================
# CREATE MODELS FOLDER
# =========================
os.makedirs("models", exist_ok=True)

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =========================
# DATASET
# =========================
dataset = ImageFolder("dataset/frames", transform=transform)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

# =========================
# MODEL (UPDATED VERSION)
# =========================
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, 2)

# =========================
# TRAINING SETUP
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# =========================
# TRAIN LOOP
# =========================
epochs = 10

for epoch in range(epochs):

    total_loss = 0

    for images, labels in loader:

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "models/video_model.pth")

print("\n✅ Training Complete!")
print("📁 Model saved at: models/video_model.pth")