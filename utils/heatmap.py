import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn

# Load Model
model = models.resnet18()
model.fc = nn.Linear(512,2)
model.load_state_dict(torch.load("models/deepfake_model.pth", map_location="cpu"))
model.eval()

# Target Layer (last conv layer)
target_layer = model.layer4[-1]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def generate_heatmap(image):

    img = Image.fromarray(image).convert("RGB")

    img_tensor = transform(img).unsqueeze(0)

    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=img_tensor)[0]

    img_np = np.array(img.resize((224,224))) / 255.0

    heatmap = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    return heatmap