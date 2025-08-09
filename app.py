import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np


def preprocess_image(pil_img):
    #Convert PIL image to OpenCV image
    img = np.array(pil_img)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  #Grayscale


    #apply Otsu's thresholding (binarization)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    #ADD noise removal, dilation/erosion HERE


    #convert back to PIL image
    return Image.fromarray(img)


st.title("📜 Gurmukhi OCR Demo with Preprocessing")

uploaded_file = st.file_uploader("Upload a Gurmukhi text image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    clean_img = preprocess_image(image)

    st.image(clean_img, caption="Preprocessed Image", use_column_width=True)

    try:
        text = pytesseract.image_to_string(clean_img, lang='pan')
    except Exception as e:
        st.error(f"OCR error: {e}")
        text = ""

    if text:
        st.subheader("Extracted Text:")
        st.text_area("Gurmukhi Text", value=text, height=200)

        text_bytes = text.encode('utf-8')
        st.download_button(
            label="Download Extracted Text as .txt",
            data=text_bytes,
            file_name="extracted_gurmukhi.txt",
            mime="text/plain"
        )
    else:
        st.warning("No text extracted. Try a clearer image.")
