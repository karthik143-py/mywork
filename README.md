# 📄 NLP Toolkit with FastAPI + Streamlit

This project is an **AI-powered NLP toolkit** that provides multiple text-processing capabilities through a **FastAPI backend** and a **Streamlit frontend**.

It supports:

- 📑 Text Summarization
- ❓ Question Answering
- 🔑 Keyword Extraction
- 🧠 Knowledge Graph Generation

The backend uses modern NLP models like **T5, spaCy, KeyBERT, SentenceTransformers, and Google Gemini**.

---

# 🏗️ Project Architecture

```
User
 │
 ▼
Streamlit UI (Frontend)
 │
 ▼
FastAPI Backend
 │
 ├── Text Summarization (T5)
 ├── Question Answering (Gemini)
 ├── Keyword Extraction (spaCy + KeyBERT)
 └── Knowledge Graph Generation (LangChain + Gemini)
```

Streamlit runs the **FastAPI server in the background** using a thread.

---

# ⚙️ Features

## 1️⃣ Text Summarization

Uses **T5-small model** from HuggingFace to summarize long text.

Endpoint:

```
POST /summarize/
```

Input:

```json
{
  "text": "Your long article here..."
}
```

Output:

```json
{
  "summary": "Short summary of the text"
}
```

---

## 2️⃣ Question Answering

Uses **Google Gemini 1.5 Flash** to answer questions based on provided text.

Endpoint:

```
POST /question/
```

Input:

```json
{
  "text": "context paragraph",
  "question": "What is the main topic?"
}
```

Output:

```json
{
  "answers": [
    {
      "answer": "Generated answer",
      "score": 1.0
    }
  ]
}
```

---

## 3️⃣ Keyword Extraction

Extracts important keywords using:

- **spaCy Named Entity Recognition**
- **KeyBERT semantic keyword extraction**

Endpoint:

```
POST /keywords/
```

Input:

```json
{
  "text": "Input text"
}
```

Output:

```json
{
  "keywords": ["keyword1", "keyword2"]
}
```

---

## 4️⃣ Knowledge Graph Generation

Creates a **visual knowledge graph** from text.

Uses:

- **LangChain Graph Transformer**
- **Google Gemini**
- **PyVis visualization**

Endpoint:

```
POST /graph/
```

Input:

```json
{
  "text": "Your article text"
}
```

Output:

A downloadable **HTML knowledge graph visualization**.

---

# 🧠 Models Used

| Component | Model |
|----------|------|
| Summarization | T5-small |
| Question Answering | Gemini 1.5 Flash |
| NER | spaCy `en_core_web_trf` |
| Keyword Extraction | KeyBERT |
| Embeddings | SentenceTransformer |
| Graph Extraction | LangChain Graph Transformer |

---

# 📦 Installation

## 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/nlp-toolkit.git
cd nlp-toolkit
```

---

## 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
fastapi
uvicorn
streamlit
torch
transformers
spacy
keybert
sentence-transformers
langchain
langchain-google-genai
langchain-experimental
pyvis
google-generativeai
python-dotenv
requests
```

---

## 3️⃣ Install spaCy model

```
python -m spacy download en_core_web_trf
```

---

## 4️⃣ Add Environment Variables

Create a `.env` file.

```
GOOGLE_API_KEY=your_gemini_api_key
```

---

# ▶️ Running the Application

Run the **Streamlit app**.

```
streamlit run app.py
```

This will:

1. Start the **FastAPI backend automatically**
2. Launch the **Streamlit UI**

Backend runs at:

```
http://127.0.0.1:8000
```

API documentation available at:

```
http://127.0.0.1:8000/docs
```

---

# 📁 Project Structure

```
project/
│
├── main.py              # FastAPI backend
├── app.py               # Streamlit frontend
├── knowledge_graph.html # Generated graph output
├── .env                 # API keys
├── requirements.txt
└── README.md
```

---

# 🔗 API Endpoints

| Endpoint | Method | Description |
|--------|--------|--------|
| `/summarize/` | POST | Generate text summary |
| `/question/` | POST | Answer questions from text |
| `/keywords/` | POST | Extract keywords |
| `/graph/` | POST | Generate knowledge graph |

---

# 📊 Knowledge Graph Visualization

The graph visualization uses **PyVis**, allowing:

- Interactive nodes
- Relationship edges
- Filtering options
- Zoom and drag support

Output file:

```
knowledge_graph.html
```

---

# 🚀 Future Improvements

- Document upload (PDF / DOCX)
- Multi-language support
- Vector database integration
- RAG pipeline
- Chat interface
- Deploy with Docker

---

# 👨‍💻 Author

Developed as an **NLP + LLM integration project** using:

- FastAPI
- Streamlit
- HuggingFace
- Google Gemini
- LangChain
