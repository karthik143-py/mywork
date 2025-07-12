import streamlit as st
import fitz  # PyMuPDF
import requests
# import torch

st.title("📄 PDF & Text Summarizer")

# === Helper: Split long text into chunks ===
def chunk_text(text, max_words=500):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

# === File Upload ===
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Extract text from PDF
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    # Break into chunks
    chunks = list(chunk_text(full_text, max_words=450))
    st.info(f"✅ PDF loaded.")

    # Show progress and summarize
    summaries = []
    progress = st.progress(0)
    for i, chunk in enumerate(chunks):
        try:
            res = requests.post("http://127.0.0.1:8000/summarize/", json={"text": chunk})
            if res.status_code == 200:
                summary = res.json().get("summary", "")
                summaries.append(f"{summary}")
            else:
                summaries.append(f"❌ API Error {res.status_code}")
        except Exception as e:
            summaries.append(f"Request failed - {e}")
        progress.progress((i + 1) / len(chunks))

    # Display combined summary
    full_summary = "\n\n".join(summaries)
    st.subheader("🧠 Final Summary")
    st.text_area("Summary Output", full_summary, height=400)

    # Optional: Download button
    st.download_button("📥 Download Summary", data=full_summary, file_name="summary.txt")
