import torch
import torch.nn as nn
import numpy as np
import glob
import matplotlib.pyplot as plt
from torchvision import models, transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
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
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ---------------- PREDICT FUNCTION FOR LIME ----------------
def predict_fn(images):
    images = torch.tensor(images).permute(0,3,1,2).float()
    images = images / 255.0
    images = images.to(device)
    with torch.no_grad():
        outputs = torch.softmax(model(images), dim=1)
    return outputs.cpu().numpy()

# ---------------- LOAD IMAGE ----------------
img_path = glob.glob("dataset/val/*/*.jpg")[0]
print("Explaining image:", img_path)

img = Image.open(img_path).convert("RGB").resize((224,224))
img_np = np.array(img)

# ---------------- LIME EXPLANATION ----------------
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    img_np,
    predict_fn,
    top_labels=1,
    hide_color=0,
    num_samples=1000
)

temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=10,
    hide_rest=False
)

# ---------------- SHOW RESULT ----------------
plt.figure(figsize=(8,6))
plt.imshow(mark_boundaries(temp/255.0, mask))
plt.title("LIME Explanation")
plt.axis('off')
plt.show()
