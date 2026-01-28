import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ---------------- DATA ----------------
val_data = datasets.ImageFolder("dataset/val", transform)
val_loader = DataLoader(val_data, batch_size=32)

# ---------------- BASELINE MODEL ----------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28,256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        return self.classifier(self.features(x))

# ---------------- LOAD MODEL ----------------
model = SimpleCNN()
model.load_state_dict(torch.load("baseline.pth", map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# ---------------- EVALUATION ----------------
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, 1).cpu()

        y_pred.extend(preds.numpy())
        y_true.extend(labels.numpy())

print("\nBaseline CNN Evaluation Results:\n")
print(classification_report(y_true, y_pred, target_names=val_data.classes))
