import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

val_data = datasets.ImageFolder("dataset/val", transform)
loader = DataLoader(val_data, batch_size=32)

# Load ResNet (change to baseline if needed)
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("resnet.pth", map_location=device, weights_only=True))
model = model.to(device)
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs,1).cpu()

        y_pred.extend(preds.numpy())
        y_true.extend(labels.numpy())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=val_data.classes,
            yticklabels=val_data.classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
