# Tumor-Detection-CNN
A machine learning model to classify brain MRI scans as **tumor-positive** or **tumor-negative** using TensorFlow/Keras and transfer learning.
## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)

---

## Overview
This project trains a CNN-based model using ResNet50 (transfer learning) to detect tumors in brain MRI scans. Key features:
- Data augmentation to handle limited medical imaging data
- Model checkpointing and early stopping
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score
- Grad-CAM visualization for model interpretability

---

## Dataset
- **Source**: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Structure**:
train/
    yes/
        img1.jpg
        img2.jpg
    no/
        img3.jpg
        ...
validation/
    yes/
    no/
test/
    yes/
    no/
- **Preprocessing**: Images resized to 224x224 pixels, normalized to [0,1].
