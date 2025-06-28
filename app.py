import streamlit as st
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite

# Load the quantized TFLite model
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model/model.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Prediction function
def predict(interpreter, img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess image
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Set input and run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # Get prediction
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return prediction

# Streamlit UI
st.set_page_config(page_title="Chest X-ray Pneumonia Detection")
st.title("Chest X-ray Pneumonia Classification")
st.write("Upload a chest X-ray image and the model will predict whether it's Pneumonia or Normal.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    interpreter = load_model()
    output = predict(interpreter, image)

    label = "Pneumonia" if output > 0.5 else "Normal"
    confidence = float(output) if output > 0.5 else 1 - float(output)

    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
