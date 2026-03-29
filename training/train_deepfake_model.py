import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch import nn, optim
from torch.utils.data import DataLoader

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Dataset
dataset = ImageFolder("dataset", transform=transform)

loader = DataLoader(dataset,batch_size=32,shuffle=True)

# Model
model = torchvision.models.resnet18(pretrained=True)

model.fc = nn.Linear(512,2)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(),lr=0.0001)

# Training
epochs = 10

for epoch in range(epochs):

    running_loss = 0

    for images,labels in loader:

        outputs = model(images)

        loss = criterion(outputs,labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print("Epoch:",epoch+1,"Loss:",running_loss)

# Save Model
torch.save(model.state_dict(),"models/deepfake_model.pth")

print("Model Training Complete")