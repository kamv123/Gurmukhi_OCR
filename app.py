import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
import re

def preprocess_image(pil_img):
    img = np.array(pil_img)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply Otsu's thresholding for binarization
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert back to PIL Image
    return Image.fromarray(img)

def keep_only_gurmukhi(text):
    # Remove any character not in Gurmukhi Unicode block or whitespace/newlines
    pattern = re.compile(r'[^\u0A00-\u0A7F\s\n]+')
    cleaned_text = pattern.sub('', text)
    return cleaned_text

st.title("📜 Gurmukhi OCR Demo with Preprocessing and Cleaning")

uploaded_file = st.file_uploader("Upload a Gurmukhi text image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    clean_img = preprocess_image(image)
    st.image(clean_img, caption="Preprocessed Image", use_column_width=True)

    try:
        raw_text = pytesseract.image_to_string(clean_img, lang='pan')
    except Exception as e:
        st.error(f"OCR error: {e}")
        raw_text = ""

    cleaned_text = keep_only_gurmukhi(raw_text)

    st.subheader("Extracted Text (Raw):")
    st.text_area("Raw OCR Output", value=raw_text, height=200)

    st.subheader("Extracted Text (Cleaned):")
    st.text_area("Cleaned Gurmukhi Text", value=cleaned_text, height=200)

    text_bytes = cleaned_text.encode('utf-8')
    st.download_button(
        label="Download Cleaned Text as .txt",
        data=text_bytes,
        file_name="cleaned_gurmukhi.txt",
        mime="text/plain"
    )
