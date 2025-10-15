import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# 1️⃣ Load the trained model
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("fruit_model.keras")  # replace with your actual file name
    return model

model = load_model()

# Define your class labels — update to match your model
class_names = [
    "Fresh Orange", "Medium Fresh Orange", "Rotten Orange",
    "Fresh Tomato", "Medium Fresh Tomato", "Rotten Tomato",
    "Fresh Carrot", "Medium Fresh Carrot", "Rotten Carrot"
]

# -------------------------------
# 2️⃣ Streamlit page setup
# -------------------------------
st.set_page_config(
    page_title="FruitSave: AI Freshness Classifier",
    page_icon="🍊",
    layout="centered"
)

st.title("🍓 **FruitSave – Fruit Freshness Classification App**")
st.markdown("""
Welcome to **FruitSave**, an AI-powered image classifier built using **Deep Learning (CNN)**.
Upload an image of a fruit below and let the model predict its **freshness level**.
""")

# -------------------------------
# 3️⃣ Image upload section
# -------------------------------
uploaded_file = st.file_uploader("📸 Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = image.resize((150, 150))  # match your training size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = np.max(prediction)

    # -------------------------------
    # 4️⃣ Display the prediction
    # -------------------------------
    st.subheader("🔍 Prediction Result")
    st.markdown(f"**Predicted Class:** {predicted_label}")
    st.markdown(f"**Confidence Level:** {confidence*100:.2f}%")

    # Optional: Add color feedback
    if "Fresh" in predicted_label:
        st.success("✅ This fruit looks Fresh!")
    elif "Medium" in predicted_label:
        st.warning("⚠️ This fruit is moderately fresh. May spoil soon.")
    else:
        st.error("❌ This fruit appears Rotten.")
else:
    st.info("Please upload an image to begin classification.")

# -------------------------------
# 5️⃣ Footer
# -------------------------------
st.markdown("""
---
👩🏽‍💻 *Developed by Group Six (Jacinta, Hasifa, Evlyne, Josephine)*  
🌍 Women in Tech AI Bootcamp 2025 – Kampala, Uganda
""")

