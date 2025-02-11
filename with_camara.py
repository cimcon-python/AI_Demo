import streamlit as st
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import os
from prompts import *
import time
from prompts import *
import re
import io

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Generative AI
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

def analyze_skin_and_recommend(image, prompt):
    try:
        response = model.generate_content([image, prompt])  # Send image directly
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def remove_coordinates_from_result(result):
    return re.sub(r'\[\d+, \d+, \d+, \d+\]', '', result).strip()

def capture_image():
    cap = cv2.VideoCapture(0)  # Open the webcam

    if not cap.isOpened():
        st.error("Could not open webcam")
        return None

    st.write("Camera will capture image in 3 seconds...")
    time.sleep(3)  # Wait for 3 seconds

    ret, frame = cap.read()  # Capture frame
    cap.release()  # Release the webcam

    if not ret:
        st.error("Failed to capture image")
        return None

    # Convert BGR to RGB (OpenCV loads images in BGR format)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)  # Convert to PIL Image

    return image

# Streamlit App
def main():
    st.title("AI Skin Analysis and Recommendation App")
    st.write("Click 'Capture Image' to take a photo using your webcam.")

    if "captured_image" not in st.session_state:
        st.session_state.captured_image = None  # Store captured image persistently

    if st.button("Capture Image"):
        st.session_state.captured_image = capture_image()

    if st.session_state.captured_image:
        st.image(st.session_state.captured_image, caption="Captured Image", use_column_width=True)

        if st.button("Analyze Skin"):
            with st.spinner("Analyzing your skin..."):
                result = analyze_skin_and_recommend(st.session_state.captured_image, analyses_prompt)
                result = remove_coordinates_from_result(result)
                st.write(result)

if __name__ == "__main__":
    main()
