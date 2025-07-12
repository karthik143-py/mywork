import streamlit as st
import fitz  # PyMuPDF
import streamlit.components.v1 as components

st.title("📄 PDF Text Extractor with Copy Feature")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
extracted_text = ""

if uploaded_file is not None:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        extracted_text += page.get_text()

    st.subheader("📄 Extracted PDF Text")
    st.text_area("PDF Content", extracted_text, height=300, key="pdf_text")

    # Copy button (inject JS)
    copy_code = f"""
    <script>
    function copyToClipboard(text) {{
        navigator.clipboard.writeText(text);
        alert("✅ Text copied to clipboard!");
    }}
    </script>
    <button onclick="copyToClipboard(`{extracted_text}`)">📋 Copy to Clipboard</button>
    """
    components.html(copy_code, height=80)
