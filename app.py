import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Page Configuration

st.set_page_config(
    page_title = "Accident Detection System",
    page_icon = "🚗",
    layout = "centered"

)

# Custom Background CSS
st.markdown("""
    <style>
        /* Main app background */
        .stApp {
            background: linear-gradient(135deg, #f0f8ff, #e6f2ff);
        }

        /* Sidebar background */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #e9f5ff, #d6ecff);
        }

        /* Title styling */
        h1 {
            color: #0078D7;
            text-shadow: 1px 1px 2px #ccc;
        }
    </style>
""", unsafe_allow_html=True)


# Button CSS
st.markdown("""
    <style>
        /* Stylish Predict Button */
        div.stButton > button {
            background: linear-gradient(90deg, #0078D7 0%, #00B4D8 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.6em 1.2em;
            font-size: 1.1em;
            font-weight: 600;
            box-shadow: 0 4px 10px rgba(0, 120, 215, 0.3);
            transition: all 0.3s ease-in-out;
        }

        div.stButton > button:hover {
            background: linear-gradient(90deg, #00B4D8 0%, #0078D7 100%);
            box-shadow: 0 6px 15px rgba(0, 120, 215, 0.5);
            transform: scale(1.05);
        }

        div.stButton > button:active {
            transform: scale(0.98);
            box-shadow: 0 2px 8px rgba(0, 120, 215, 0.4);
        }
    </style>
""", unsafe_allow_html=True)


# Load Model

@st.cache_resource
def load_accident_model():
    model = load_model("final_accident_detection_model.keras")
    return model

model = load_accident_model()

# Class Labels

class_names = ["Accident", "Non Accident"]

# Sidebar

st.sidebar.title("About Project")

st.sidebar.info(
    """
### Accident Detection System

This application uses a deep learning model to classify CCTV images into:

- 🚨 Accident
- ✅ Non Accident

"""
)

# Main Title

st.title("🚗 Accident Detection using Deep Learning")

st.markdown("---")

st.write(
"""
Upload a CCTV road image and click **Predict** to determine whether an accident has occurred.
"""
)

# Image Upload

uploaded_file = st.file_uploader(
    "Choose an Image", type = ["jpg", "jpeg", "png"]
)

# Prediction

if uploaded_file is not None:
    image= Image.open(uploaded_file)

    st.markdown("<h4 style='color:#0078D7;'>Uploaded Image Preview</h4>", unsafe_allow_html=True)
    st.image(image, caption="CCTV Frame", use_container_width=True)
    st.markdown("<hr style='border:1px solid #0078D7;'>", unsafe_allow_html=True)


    if st.button("Predict"):
        img = image.resize((224,224))
        img_array = np.array(img)
        img_array = img_array.astype("float32")
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        probability = prediction[0][0]

        if probability < 0.5:

            predicted_class = "🚨 Accident"
            confidence = (1-probability)*100

        else:

            predicted_class = "✅ Non Accident"
            confidence = probability*100

        st.markdown("---")

        st.markdown("""
    <div style='background-color:#312C85; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.1); padding:20px; margin-top:20px;'>
        <h3 style='color:#FDFEFE; text-align:center;'>Prediction Result</h3>
    </div>
""", unsafe_allow_html=True)

        st.markdown(f"""
    <div style='background-color:#e8f5e9; border-left:6px solid #4CAF50; border-radius:8px; padding:12px; margin-top:10px;'>
        <h4 style='color:#2e7d32;'>Prediction : {predicted_class}</h4>
    </div>
""", unsafe_allow_html=True)

        st.markdown(f"""
    <div style='background-color:#e3f2fd; border-left:6px solid #2196F3; border-radius:8px; padding:12px; margin-top:10px;'>
        <h4 style='color:#1565c0;'>Confidence : {confidence:.2f}%</h4>
    </div>
""", unsafe_allow_html=True)


        