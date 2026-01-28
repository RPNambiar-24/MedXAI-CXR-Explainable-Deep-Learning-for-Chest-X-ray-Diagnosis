# 🩺 MedXAI-CXR: Explainable Deep Learning for Chest X-ray Diagnosis

## 🌟 Overview

**MedXAI-CXR** is an explainable deep learning system for automated chest X-ray diagnosis. The goal is not only to achieve high classification accuracy but also to understand **why** the model makes decisions — a key requirement for trustworthy medical AI.

The system classifies X-rays into:

- 🦠 COVID-19  
- 🫁 Pneumonia  
- ✅ Normal  

We compare a **baseline CNN** with a **transfer learning ResNet50 model** and analyze model behavior using **Grad-CAM, LIME, and SHAP**.

---

## 🎯 Objectives

- Develop a deep learning model for chest X-ray classification  
- Compare baseline vs transfer learning performance  
- Evaluate model trustworthiness using explainability methods  
- Identify which XAI technique provides the most clinically meaningful insight  

---

## 📂 Dataset

This project uses the **Chest X-ray COVID-19 & Pneumonia Dataset** from Kaggle:

🔗 https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia

### 📖 Context

COVID-19 is an infectious disease caused by SARS-CoV-2. While RT-PCR is the standard diagnostic method, chest X-ray (CXR) imaging provides rapid and widely accessible screening support. AI-based analysis of CXR images can assist in early detection and diagnosis.

### 📦 Dataset Structure

The dataset contains **6432 chest X-ray images** organized as:

```
dataset/  
├── train/  
│ ├── COVID19/  
│ ├── NORMAL/  
│ └── PNEUMONIA/  
└── val/  
│ ├── COVID19/  
│ ├── NORMAL/  
│ └── PNEUMONIA/  
```

- 3 classes: COVID-19, Pneumonia, Normal  
- Test data ≈ 20% of total images  

### 🙏 Acknowledgements

Images are collected from publicly available sources. Please credit original contributors when using the dataset:

- https://github.com/ieee8023/covid-chestxray-dataset  
- https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia  
- https://github.com/agchung  

---

## ⚙️ Methodology

### 🧠 Models Used

| Model | Description |
|------|-------------|
| Baseline CNN | Custom shallow convolutional neural network |
| ResNet50 | Transfer learning model with pretrained ImageNet weights |

### 🏋️ Training Setup

- Loss Function: CrossEntropyLoss  
- Optimizer: Adam  
- Learning Rate: 0.0001  
- Epochs: 5  
- Batch Size: 32  

### 📏 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## 📊 Results

### 🧮 Model Performance

| Model | Accuracy | Macro F1 | COVID Recall | Normal Recall | Pneumonia Recall |
|-------|----------|----------|--------------|---------------|------------------|
| Baseline CNN | 0.96 | 0.96 | 0.96 | 0.93 | 0.98 |
| ResNet50 | **0.97** | **0.97** | **1.00** | 0.92 | 0.98 |

Transfer learning improved diagnostic sensitivity, especially for COVID detection.

---

### 📈 Training Loss Comparison

ResNet50 converged faster and achieved lower final loss compared to the baseline CNN, demonstrating the benefit of pretrained features.

---

### 📉 Confusion Matrix Insight

Most misclassifications occurred between Normal and Pneumonia, reflecting clinically subtle differences. COVID detection remained highly reliable.

---

## 🔍 Explainability Analysis

| Method | Strength | Limitation |
|--------|----------|-----------|
| Grad-CAM | Strong anatomical alignment | Coarse spatial resolution |
| LIME | Local decision reasoning | Superpixel boundary leakage |
| SHAP | Feature contribution precision | Less intuitive spatial mapping |

Grad-CAM provided the most clinically interpretable explanations.

---

## 🧠 Key Findings

- Transfer learning improves medical image classification performance  
- High accuracy does not guarantee trustworthy AI  
- Spatial attention methods (Grad-CAM) are most suitable for medical imaging interpretability  
- Multi-method XAI provides stronger validation of model reliability  

---

## 🗂️ Project Structure

train_baseline.py  
train_resnet.py  
evaluate.py  
evaluate_baseline.py  
gradcam.py  
lime_explain.py  
shap_explain.py  
confusion_matrix.py  
plot_loss.py  

---

## ▶️ How to Run

### Train Models

python train_baseline.py  
python train_resnet.py  

### Evaluate

python evaluate_baseline.py  
python evaluate.py  

### Explainability

python gradcam.py  
python lime_explain.py  
python shap_explain.py  

---

## 🏁 Conclusion

Combining **transfer learning with explainable AI** produces a highly accurate and interpretable medical imaging system. Grad-CAM emerged as the most clinically meaningful explanation method, supporting its use in AI-assisted radiology.

---

## 🚀 Future Work

- Incorporate radiologist validation  
- Use segmentation-guided explanations  
- Extend to other imaging modalities (CT, MRI)  
- Explore transformer-based medical models  

---

## 👨‍💻 Author

**Rishab P Nambiar**  
Integrated MSc Data Science Student  
Research Focus: AI, Medical Imaging, Explainable AI
