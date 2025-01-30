import streamlit as st
from PIL import Image, ImageDraw
import google.generativeai as genai
from dotenv import load_dotenv
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import *
import os
import warnings

st.set_option('deprecation.showPyplotGlobalUse', False)  # Example for other warnings
warnings.filterwarnings("ignore", message="The use_column_width parameter has been deprecated")

load_dotenv()

# Configure Generative AI
api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-pro")
# chat_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5, max_output_tokens=8192)
# Function to analyze skin and provide recommendations
def analyze_skin_and_recommend(image, prompt):
    try:
        # Generate content using the AI model
        response = model.generate_content([image, prompt])
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def remove_coordinates_from_result(result):
    cleaned_result = re.sub(r'\[\d+, \d+, \d+, \d+\]', '', result)
    return cleaned_result.strip()

# Streamlit App
def main():
    st.title("AI Skin Analysis and Recommendation App")
    st.write("Upload an image of your face, and let our AI provide a detailed skin analysis, scores for specific concerns, and personalized skincare recommendations.")

    # File uploader for image input
    uploaded_image = st.file_uploader("Upload an image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        width, height = image.size  # Get image dimensions
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Skin"):
            with st.spinner("Analyzing your skin..."):
                result = analyze_skin_and_recommend(image, analyses_prompt).replace('json','').replace('```','')
                try:
                    st.write(result)
                except json.JSONDecodeError as e:
                    st.error("Failed to parse the AI response. Please try again.",e)


if __name__ == "__main__":
    main()
