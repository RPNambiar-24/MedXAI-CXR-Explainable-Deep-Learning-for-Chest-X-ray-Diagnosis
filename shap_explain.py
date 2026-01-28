import torch
import torch.nn as nn
import shap
import numpy as np
import glob
from torchvision import models, transforms, datasets
from PIL import Image

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("resnet.pth", map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# ---------------- BACKGROUND ----------------
train_data = datasets.ImageFolder("dataset/train", transform)
loader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True)

background, _ = next(iter(loader))
background = background.to(device)

# Use GradientExplainer (stable for CNNs)
explainer = shap.GradientExplainer(model, background)

# ---------------- IMAGE ----------------
img_path = glob.glob("dataset/val/*/*.jpg")[0]
print("Explaining image:", img_path)

img = transform(Image.open(img_path)).unsqueeze(0).to(device)

# ---------------- SHAP VALUES ----------------
shap_values = explainer.shap_values(img)

# 🔥 FIX SHAPE: (C,H,W) → (H,W,C)
img_np = img.cpu().numpy().transpose(0,2,3,1)

# shap_values is list per class → transpose each
shap_values = [sv.transpose(0,2,3,1) for sv in shap_values]

# ---------------- PLOT ----------------
shap.image_plot(shap_values, img_np)
