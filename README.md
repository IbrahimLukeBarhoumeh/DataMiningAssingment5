# Underwater Acoustic Ship Classification  
*Data Mining Assignment 5*

This project leverages Deep Learning to classify ships based on underwater acoustic signals using spectrogram analysis. The goal is to distinguish between different vessel types by interpreting frequency-time patterns from underwater audio recordings.

---

## 🔍 Project Overview

The dataset consists of underwater recordings of three distinct ship types:
- **KaiYuan**
- **SpeedBoat**
- **UUV**

Raw `.wav` audio is transformed into Mel spectrograms using the Short-Time Fourier Transform (STFT), allowing models to interpret time-frequency features as 2D images. Both traditional **Convolutional Neural Networks (CNNs)** and modern **Transformer-based models** are used and compared.

---

## 🎯 Project Objectives

- Analyze and understand the structure and balance of the dataset
- Convert underwater audio to spectrograms using STFT
- Normalize and resize inputs to a fixed shape
- Implement and evaluate two architectures: CNN and Transformer
- Compare model performance across key classification metrics
- Identify challenges, limitations, and recommendations for future work

---

## ✅ Task Breakdown

### **Task 1: Data Understanding & Preprocessing**
- Explored class distributions and audio durations
- Generated Mel spectrograms from raw audio
- Normalized and resized images to 128×128
- Saved preprocessed data as `.npy` for efficiency
- Visualized spectrogram samples from each ship type

### **Task 2: CNN Model Design and Training**
- Developed a multi-layer CNN using Conv → MaxPool → Dropout layers
- Used early stopping to prevent overfitting
- Achieved strong classification performance:
  - **Accuracy**: 76%
  - **Best Performing Class**: SpeedBoat (F1 = 0.86)
  - **Most Challenging Class**: KaiYuan (F1 = 0.61)

### **Task 3: Transformer Model Design and Training**
- Implemented a patch-based Vision Transformer (ViT) architecture
- Applied learnable positional embeddings and multi-head attention
- Less accurate than CNN due to smaller dataset size:
  - **Accuracy**: 67%
  - **Best Performing Class**: SpeedBoat (F1 = 0.81)
  - **Most Challenging Class**: KaiYuan (F1 = 0.32)

---

## 📊 Final Evaluation

| Model         | Accuracy | Best F1 Score     | Weakest F1 Score     |
|---------------|----------|------------------|-----------------------|
| **CNN**        | 76%      | SpeedBoat (0.86)  | KaiYuan (0.61)        |
| **Transformer**| 67%      | SpeedBoat (0.81)  | KaiYuan (0.32)        |

---

## 📈 Key Insights

- **Spectrograms** are highly effective in transforming audio into usable visual inputs for classification tasks.
- **CNNs** performed better on this task due to their strength in capturing local spatial features and their lower dependency on large datasets.
- **Transformers**, while powerful for long-range dependencies, struggled without extensive data and tuning.
- Both models performed well on SpeedBoat samples due to clearer acoustic signatures and more balanced data.
- **KaiYuan was the weakest class across both models**, highlighting the need for class balancing or augmentation.

---

## 🔮 Recommendations for Future Work

- **Data Augmentation**: Apply time stretching, pitch shifting, or Gaussian noise to improve generalization and balance underrepresented classes like KaiYuan.
- **Transfer Learning**: Use pretrained audio-based models (e.g., AudioSet, YAMNet) for feature extraction and fine-tuning.
- **Hybrid Models**: Combine CNN feature extractors with Transformer encoders to benefit from both local and global context.
- **Larger Dataset**: Collect more samples for each class to reduce bias and improve Transformer training.
- **Audio Preprocessing**: Apply noise reduction, bandpass filtering, or frequency normalization before spectrogram conversion.

---



## 📌 Author

**Project by:** Ibrahim Luke Barhoumeh  
**Course:** GEA 2000 – World Geography  
**Institution:** Florida Atlantic University

