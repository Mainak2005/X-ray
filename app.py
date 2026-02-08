import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page config
st.set_page_config(page_title="Pneumonia Detection", page_icon="🩺", layout="centered")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    **AI Pneumonia Detection App**\n
    - Model: ResNet50 Transfer Learning
    - Input: Chest X‑ray Image
    - Output: Pneumonia / Normal prediction

    Upload an X‑ray image to test the model.
    """
)

# Load model with caching (faster reload)
@st.cache_resource
def load_my_model():
    return load_model("xray_resnet50_final.h5", compile=False)

model = load_my_model()

IMG_SIZE = 224

st.title("🩺 Pneumonia Detection from Chest X‑Ray")
st.write("Upload a chest X-ray image and the AI model will predict pneumonia presence.")

uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]

    st.subheader("Prediction Result")

    if prediction > 0.5:
        st.error("⚠ Pneumonia Detected")
    else:
        st.success("✅ Normal")

    st.warning("""
    ⚠ Important Precautions:
    - This AI prediction is NOT a medical diagnosis.
    - Always consult a qualified doctor or radiologist.
    - Image quality, angle, and dataset limitations may affect results.
    - Do not make health decisions based solely on this tool.
    """)

    st.caption("Educational AI demo only — not for clinical use.")
