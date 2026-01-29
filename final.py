import streamlit as st
from google import genai
from PIL import Image
import tf_keras as keras
import tensorflow as tf
import numpy as np

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

st.title("🌱 Green Campus Waste Management System 🌱")
model = keras.models.load_model("keras_model.h5")

st.write("Capture an image of waste to identify and manage it properly.")

captured_image = st.camera_input("📸 Capture waste image")

if captured_image:
    image = Image.open(captured_image).resize((224,224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    st.image(image, caption="Captured Waste Image", use_column_width=True)

    if st.button("♻️ Analyze Waste"):
        with st.spinner("Analyzing waste ..."):
            prediction = model.predict(image_array)

        labels = ["Dry(Plastic/Metal)","Wet (Organic)","Paper","Hazardous (E-waste/Masks)"]
        class_index = np.argmax(prediction)
        st.success(f"This looks like: {labels[class_index]}")

        with st.spinner("Analyzing waste using AI..."):
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=["Tell me in which bin should this waste be discarded in one sentence red/green/blue:{labels[class_index]} also add bin colour emoji", image]
            )
        st.success("✅ Waste analysis completed successfully!")
        st.write("### 🧾 AI Suggestion:")
        st.write(response.text)
