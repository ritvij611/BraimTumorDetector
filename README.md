# Brain Tumor Detection using Convolutional Neural Networks

A deep learning project that classifies brain MRI images to detect and categorize different types of brain tumors using a Convolutional Neural Network (CNN) architecture.

## 🧠 Project Overview

This project implements a CNN-based classifier to automatically detect and classify brain tumors from MRI scan images. The model can distinguish between four different classes of brain conditions, making it a valuable tool for medical imaging analysis and diagnosis assistance.

## 🎯 Objective

Develop an automated system that can:
- Detect the presence of brain tumors in MRI scans
- Classify different types of brain tumors
- Assist medical professionals in preliminary diagnosis
- Provide accurate and reliable predictions with 77% accuracy

## 📊 Dataset

**Source**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

**Dataset Characteristics**:
- **Classes**: 4 different categories
- **Image Format**: MRI scans
- **Input Shape**: 224 × 224 × 3 (RGB images)
- **Data Type**: Medical imaging data

## 🏗️ Model Architecture

The CNN model follows a sequential architecture with the following key components:

### Network Structure:
```
Input Layer: (224, 224, 3)
↓
Conv2D (32 filters, 3×3) + BatchNorm + ReLU
Conv2D (32 filters, 3×3) + BatchNorm + ReLU + MaxPool2D + Dropout(0.25)
↓
Conv2D (64 filters, 3×3) + BatchNorm + ReLU
Conv2D (64 filters, 3×3) + BatchNorm + ReLU + MaxPool2D + Dropout(0.25)
↓
Conv2D (128 filters, 3×3) + BatchNorm + ReLU
Conv2D (128 filters, 3×3) + BatchNorm + ReLU + MaxPool2D + Dropout(0.4)
↓
GlobalAveragePooling2D
Dense (128 units) + ReLU + Dropout(0.5)
Dense (4 units) + Softmax
```

### Key Features:
- **Convolutional Layers**: Progressive filter increase (32 → 64 → 128)
- **Batch Normalization**: Applied after each convolution for stable training
- **Dropout Regularization**: Prevents overfitting (0.25, 0.4, 0.5)
- **Global Average Pooling**: Reduces parameters compared to flatten
- **Activation Functions**: ReLU for hidden layers, Softmax for output

## 📈 Model Performance

- **Training Accuracy**: 77%
- **Architecture**: Sequential CNN with 6 convolutional layers
- **Total Parameters**: Optimized for medical imaging classification
- **Regularization**: Multiple dropout layers and batch normalization

## 🛠️ Implementation Details

### Dependencies:
```python
tensorflow
keras
numpy
matplotlib
pandas
scikit-learn
```


## 🚀 Getting Started

### Prerequisites:
1. Python 3.7+
2. TensorFlow 2.x
3. Required Python packages (see dependencies)

### Installation:
```bash
pip install tensorflow keras numpy matplotlib pandas scikit-learn
```

### Usage:
1. Download the dataset from Kaggle
2. Preprocess the MRI images (resize to 224×224)
3. Split data into training/validation/test sets
4. Train the model using the provided architecture
5. Evaluate performance and make predictions

## 🔬 Medical Applications

This model can assist in:
- **Preliminary Screening**: Quick assessment of MRI scans
- **Second Opinion**: Supporting radiologists' diagnoses
- **Research**: Analyzing large datasets of brain imaging
- **Education**: Training medical students on tumor recognition


## 📊 Future Improvements

Potential enhancements to increase accuracy:
- Data augmentation techniques
- Transfer learning with pre-trained models (ResNet, VGG, EfficientNet)
- Ensemble methods
- Advanced preprocessing techniques
- Larger and more diverse datasets
- Cross-validation strategies



---

**Note**: This project demonstrates the application of deep learning in medical imaging. The 77% accuracy shows promising results, with room for improvement through advanced techniques and larger datasets.