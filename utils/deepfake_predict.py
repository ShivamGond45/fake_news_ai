import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torch import nn
import os
from utils.face_detection import extract_face
model = models.resnet18()
model.fc = nn.Linear(512,2)

# MODEL LOAD SAFE
model_path = "models/deepfake_model.pth"

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("Model loaded successfully")
    except:
        print("Model file corrupted. Using random weights.")
else:
    print("Model file not found. Using random weights.")

model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict_image(image):

    face = extract_face(image)

    img = Image.fromarray(face).convert("RGB")

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    prob = torch.softmax(output, dim=1)

    fake_prob = prob[0][0].item()

    if fake_prob > 0.6:
        return "FAKE", fake_prob
    else:
        return "REAL", 1 - fake_prob