# 🫁 Pneumonia Detection from Chest X-ray using Deep Learning

A web-based application that uses a **Convolutional Neural Network (CNN)** model to detect **Pneumonia** from chest X-ray images. Built with TensorFlow, converted to TFLite for lightweight deployment, and served using **Streamlit**.

## 🚀 Live Demo

👉 [Click here to try the app](https://chestxray-<your-app-id>.streamlit.app/)  
*(Replace with your actual deployed Streamlit Cloud link)*

---

## 📌 Features

- Upload chest X-ray images directly through the browser
- Real-time prediction: "Pneumonia" or "Normal"
- Model optimized with TensorFlow Lite for efficient inference
- Streamlit-powered user interface
- Fully deployed and accessible online

---

## 🧠 Model Architecture

- **Input Shape**: `(224, 224, 3)`
- **CNN Layers**: Depthwise Separable Convolutions + BatchNorm
- **Pooling**: MaxPooling
- **Fully Connected Layers**
- **Output**: Binary classification (Sigmoid activation)

---

## 📂 Project Structure

chest_xray/
│
├── app.py # Streamlit application
├── requirements.txt # Dependencies for deployment
├── .gitignore
├── model/
│ └── model.tflite # Trained & quantized TFLite model


---

## 🖼 Sample Predictions

| Image | Prediction |
|-------|------------|
| ![](sample_normal.png) | ✅ Normal |
| ![](sample_pneumonia.png) | ❗ Pneumonia |

---

## 🧪 How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/GD-5002/chest_xray.git
   cd chest_xray
