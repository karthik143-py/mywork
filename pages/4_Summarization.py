import streamlit as st
import fitz  # PyMuPDF
import requests

st.title("📄 Text Summarization (PDF or Text)")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

extracted_text = ""

if uploaded_file is not None:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        extracted_text += page.get_text()
    
    st.subheader("Extracted PDF Text")
    st.text_area("PDF Content", extracted_text, height=300)

# Text area for manual input
text_input = st.text_area("Or paste your text here:")

# Choose which text to summarize: PDF or Manual
final_text = extracted_text if extracted_text.strip() != "" else text_input

if st.button("Summarize"):
    if final_text.strip() == "":
        st.warning("❗ Please upload a PDF or enter text.")
    else:
        with st.spinner("Generating summary..."):
            try:
                response = requests.post("http://127.0.0.1:8000/summarize/", json={"text": final_text})
                if response.status_code == 200:
                    summary = response.json().get("summary", "No summary returned.")
                    st.success("✅ Summary generated successfully!")
                    st.subheader("Summary")
                    st.write(summary)
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Request failed: {e}")
