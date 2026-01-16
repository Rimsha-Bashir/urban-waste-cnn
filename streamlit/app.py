import streamlit as st
import base64
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_URL = os.getenv("API_URL")

st.set_page_config(page_title="Urban Waste Classifier", layout="centered")

st.title("Urban Waste Classifier")
st.write("Upload an image and get a prediction from your AWS Lambda model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img_bytes = uploaded_file.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    if st.button("Predict"):
        with st.spinner("Sending to model…"):
            try:
                response = requests.post(API_URL, json={"image": img_b64})
                data = response.json()

                if "prediction" in data:
                    st.success(f"Prediction: **{data['prediction']}**")
                    st.info(f"Confidence: **{data['confidence']:.4f}**")
                else:
                    st.error(f"Error: {data.get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"Request failed: {e}")
