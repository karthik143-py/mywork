import streamlit as st
import requests

FASTAPI_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Knowledge Graph Generator", layout="wide")
st.title("📚 Knowledge Graph Generator")

input_text = st.text_area("Enter your text for graph generation:", height=300)

if st.button("Generate Knowledge Graph"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating graph..."):
            response = requests.post(f"{FASTAPI_URL}/graph/", json={"text": input_text})

            if response.status_code == 200:
                with open("knowledge_graph.html", "wb") as f:
                    f.write(response.content)

                with open("knowledge_graph.html", "rb") as f:
                    st.download_button(label="📥 Download Knowledge Graph",
                                       data=f,
                                       file_name="knowledge_graph.html",
                                       mime="text/html")
                st.success("Graph generated successfully!")
            else:
                st.error("Failed to generate graph. Please try again.")
