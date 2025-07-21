import tempfile
import requests
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Load the model
url = "https://drive.google.com/uc?export=download&id=1ivIb-RMuAL31mabUSuT7nA17uyY4mkj1"

response = requests.get(url)
response.raise_for_status()

# Save it to a temporary file
with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
    temp_file.write(response.content)
    temp_model_path = temp_file.name
model = tf.keras.models.load_model(temp_model_path)

# Class labels (same order as your dataset folders)
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Streamlit UI
st.set_page_config(page_title="🌸 Flower Classifier", layout="centered")
st.title("🌼 Flower Image Classifier")
st.write("Upload a flower image and I'll try to predict its type!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    # Show prediction
    st.success(f"🌸 I'm {100 * np.max(predictions):.2f}% sure this is a **{predicted_class}**!")
