import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch import nn, optim

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = ImageFolder("dataset", transform=transform)

loader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True)

model = torchvision.models.resnet18(pretrained=True)

model.fc = nn.Linear(512,2)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=0.0001)

for epoch in range(5):

    for images,labels in loader:

        outputs = model(images)

        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch",epoch,"Loss",loss.item())

torch.save(model.state_dict(),"models/deepfake_model.pth")