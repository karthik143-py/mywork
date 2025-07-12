import streamlit as st
import fitz  # PyMuPDF
import requests

st.title("🔑 Keyword Extraction (PDF or Text)")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

extracted_text = ""

if uploaded_file is not None:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        extracted_text += page.get_text()
    
    # st.subheader("Extracted PDF Text")
    # st.text_area("PDF Content", extracted_text, height=300)

# Manual text input
# manual_text = st.text_area("Or paste your text here:")

# Choose text source
final_text = extracted_text

# Button to extract keywords
if st.button("Extract Keywords"):
    with st.spinner("Extracting keywords..."):
        if final_text.strip() == "":
            st.warning("❗ Please upload a PDF or enter text.")
        else:
            with st.spinner("Extracting keywords..."):
                try:
                    response = requests.post("http://127.0.0.1:8000/keywords/", json={"text": final_text})
                    if response.status_code == 200:
                        keywords = response.json().get("keywords", [])
                        if not keywords:
                            st.info("No keywords found.")
                        else:
                            st.success("✅ Extracted Keywords:")
                            st.write(", ".join(keywords))
                    else:
                        st.error(f"Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Request failed: {e}")
