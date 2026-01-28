# Explainable AI for Chest X-ray Disease Classification

## Overview

This project investigates the use of **deep learning with explainable AI (XAI)** for multi-class chest X-ray diagnosis. The goal is not only to achieve high classification accuracy but also to understand **why** the model makes decisions, which is critical for clinical trust.

The system classifies X-rays into:

- COVID-19  
- Pneumonia  
- Normal  

We compare a **baseline CNN** with a **transfer learning ResNet50 model** and analyze model behavior using **Grad-CAM, LIME, and SHAP**.

---

## Objectives

- Develop a deep learning model for chest X-ray classification  
- Compare baseline vs transfer learning performance  
- Evaluate model trustworthiness using explainability methods  
- Identify which XAI technique provides the most clinically meaningful insight  

---

## Dataset

The dataset contains chest X-ray images organized into:

dataset/  
├── train/  
│ ├── COVID19/  
│ ├── NORMAL/  
│ └── PNEUMONIA/  
└── val/  
&nbsp;&nbsp;&nbsp;&nbsp;├── COVID19/  
&nbsp;&nbsp;&nbsp;&nbsp;├── NORMAL/  
&nbsp;&nbsp;&nbsp;&nbsp;└── PNEUMONIA/  

Images are resized to **224×224** and normalized for deep learning input.

---

## Methodology

### Models Used

| Model | Description |
|------|-------------|
| Baseline CNN | Custom shallow convolutional neural network |
| ResNet50 | Transfer learning model with pretrained ImageNet weights |

### Training Setup

- Loss Function: CrossEntropyLoss  
- Optimizer: Adam  
- Learning Rate: 0.0001  
- Epochs: 5  
- Batch Size: 32  

### Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## Results

### Model Performance

| Model | Accuracy | Macro F1 | COVID Recall | Normal Recall | Pneumonia Recall |
|-------|----------|----------|--------------|---------------|------------------|
| Baseline CNN | 0.96 | 0.96 | 0.96 | 0.93 | 0.98 |
| ResNet50 | **0.97** | **0.97** | **1.00** | 0.92 | 0.98 |

Transfer learning improved diagnostic sensitivity, especially for COVID detection.

---

### Training Loss Comparison

ResNet50 converged faster and achieved lower final loss compared to the baseline CNN, demonstrating the benefit of pretrained features.

---

### Confusion Matrix Insight

Most misclassifications occurred between Normal and Pneumonia, reflecting clinically subtle differences. COVID detection remained highly reliable.

---

## Explainability Analysis

| Method | Strength | Limitation |
|--------|----------|-----------|
| Grad-CAM | Strong anatomical alignment | Coarse spatial resolution |
| LIME | Local decision reasoning | Superpixel boundary leakage |
| SHAP | Feature contribution precision | Less intuitive spatial mapping |

Grad-CAM provided the most clinically interpretable explanations.

---

## Key Findings

- Transfer learning improves medical image classification performance  
- High accuracy does not guarantee trustworthy AI  
- Spatial attention methods (Grad-CAM) are most suitable for medical imaging interpretability  
- Multi-method XAI provides stronger validation of model reliability  

---

## Project Structure

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

## How to Run

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

## Conclusion

The study demonstrates that combining **transfer learning with explainable AI** produces a highly accurate and interpretable medical imaging system. Grad-CAM emerged as the most clinically meaningful explanation method, supporting its use in AI-assisted radiology.

---

## Future Work

- Incorporate radiologist validation  
- Use segmentation-guided explanations  
- Extend to other imaging modalities (CT, MRI)  
- Explore transformer-based medical models  

---

## Author

Rishab P Nambiar  
Integrated MSc Data Science Student  
Research Focus: AI, Medical Imaging, Explainable AI
