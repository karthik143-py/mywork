import streamlit as st
import fitz  # PyMuPDF
import requests
st.title("📄 Text Extractor(PDF)")
# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
extracted_text = ""
if uploaded_file is not None:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        extracted_text += page.get_text()
    st.subheader("Extracted PDF Text")
    st.text_area("PDF Content", extracted_text, height=300)
