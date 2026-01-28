import torch
import torch.nn as nn
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from torchvision import models, transforms
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
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ---------------- GRADCAM CLASS ----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x, class_idx):
        output = self.model(x)
        self.model.zero_grad()
        output[:, class_idx].backward()

        weights = self.gradients.mean(dim=[2,3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).cpu().detach().numpy()

        cam = cv2.resize(cam, (224,224))
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

# ---------------- PICK IMAGE FROM DATASET ----------------
img_path = glob.glob("dataset/val/*/*.jpg")[0]
print("Explaining image:", img_path)

img_pil = Image.open(img_path).convert("RGB")
img_tensor = transform(img_pil).unsqueeze(0).to(device)

# ---------------- PREDICTION ----------------
output = model(img_tensor)
pred_class = torch.argmax(output, 1).item()
print("Predicted class index:", pred_class)

# ---------------- GENERATE CAM ----------------
gradcam = GradCAM(model, model.layer4[-1])
cam = gradcam.generate(img_tensor, pred_class)

# ---------------- OVERLAY HEATMAP ----------------
img_np = np.array(img_pil.resize((224,224)))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

# ---------------- SHOW RESULT ----------------
plt.figure(figsize=(8,6))
plt.imshow(overlay)
plt.title("Grad-CAM Explanation")
plt.axis('off')
plt.show()
