import streamlit as st
import fitz  # PyMuPDF
import requests

st.title("❓ Question Answering (PDF or Text)")

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

# Question input
question = st.text_input("Your Question:")

# Button to get answers
if st.button("Get Answers"):
    if final_text.strip() == "" or question.strip() == "":
        st.warning("❗ Please upload a PDF or enter text, and provide a question.")
    else:
        with st.spinner("Searching for answers..."):
            try:
                response = requests.post("http://127.0.0.1:8000/question/", json={"text": final_text, "question": question})
                if response.status_code == 200:
                    answers = response.json().get("answers", [])
                    if not answers:
                        st.info("No answers found.")
                    else:
                        st.success("✅ Answers:")
                        for ans in answers:
                            st.write(f"**{ans['answer']}**  _(Confidence: {ans['score']:.2f})_")
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Request failed: {e}")
