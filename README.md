# 🥔 Potato Disease Detection

A machine learning project for detecting diseases in potato leaves using image classification techniques. This system helps farmers and agricultural professionals quickly identify and classify common potato leaf diseases to take timely action.

---

## 🔍 Overview

Potatoes are vulnerable to several leaf diseases that can severely affect crop yield. This project uses deep learning to classify potato leaf images into the following categories:

- **Early Blight**
- **Late Blight**
- **Healthy**

The system takes a leaf image as input and predicts the disease category, enabling early diagnosis and better crop management.

---

## 📂 Dataset

The dataset used for this project is the [**PlantVillage Dataset**](https://www.kaggle.com/datasets/emmarex/plantdisease), specifically filtered for:

- `Potato___Early_blight`
- `Potato___Late_blight`
- `Potato___Healthy`

---

## 🧠 Model Details

- **Model Type:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras *(modify if using PyTorch or others)*
- **Image Preprocessing:** Resized to 256x256 pixels, normalized pixel values
- **Train/Validation Split:** 80% training, 20% validation

---

## 🛠️ Features

- Upload a potato leaf image and get instant disease prediction
- Trained on real-world, high-quality image data
- Can be deployed to web or mobile apps
- Lightweight and scalable for practical use

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/potato-disease-detection.git
cd potato-disease-detection

### 2. Install Dependencies
```bash
Copy
Edit
pip install -r requirements.txt

### 3. Train the Model
```bash
Copy
Edit
python train.py

### 4. Make Predictions
```bash
Copy
Edit
python predict.py --image path_to_leaf_image.jpg
