# app.py

import streamlit as st
import threading
import uvicorn
from fastapi import FastAPI
from main import app as fastapi_app  # this is your FastAPI backend

# ✅ Start FastAPI backend in background
def run_fastapi():
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000)

threading.Thread(target=run_fastapi, daemon=True).start()

# ✅ Streamlit UI
st.set_page_config(page_title="Streamlit + FastAPI")
st.title("Welcome to the Home Page")

# Example API call to backend (optional)
import requests
try:
    r = requests.get("http://127.0.0.1:8000/")
    # st.success(f"FastAPI response: {r.text}")
except:
    st.warning("Waiting for FastAPI backend to start...")
