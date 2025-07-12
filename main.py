# from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File

from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import pipeline
import spacy
from keybert import KeyBERT
# import networkx as nx
import google.generativeai as genai
# import matplotlib.pyplot as plt
# import io
# import base64

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from pyvis.network import Network
# import asyncio
import uvicorn
import os
from sentence_transformers import SentenceTransformer, util
load_dotenv()
app = FastAPI()

# Models
api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key)
graph_transformer = LLMGraphTransformer(llm=llm)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_model = pipeline('question-answering')
nlp = spacy.load("en_core_web_trf")
kw_model = KeyBERT(model='all-mpnet-base-v2')
embedder = SentenceTransformer('all-MiniLM-L6-v2')
genai.configure(api_key="AIzaSyBhR2rNPTlAC6zyIVWPKKUMFfP54ZZ427A")


class TextInput(BaseModel):
    text: str

class QAInput(BaseModel):
    text: str
    question: str

@app.post("/summarize/")
async def summarize(data: TextInput):
    word_count = len(data.text.split())
    min_length = min(max(int(word_count * 0.3), 30), 150)    # never too big
    max_length = min(max(int(word_count * 0.5), 60), 250)  
    result = summarizer(data.text, min_length=min_length,max_length=max_length, do_sample=False)
    return {"summary": result[0]['summary_text']}

@app.post("/question/")
async def question(data: QAInput):
    model = genai.GenerativeModel("models/gemini-1.5-flash") 
    prompt = f"""You are an expert reader. Carefully read the following text and answer the question accurately.Text:{data.text},Question:{data.question}"""

    response = model.generate_content(prompt)
    answer = response.text.strip()

    return {"answers": [{"answer": answer, "score": 1.0}]}

@app.post("/keywords/")
async def keywords(data: TextInput):
    doc = nlp(data.text)
    keyphrases = list(set([ent.text.lower() for ent in doc.ents] +
                          [kw[0].lower() for kw in kw_model.extract_keywords(data.text, keyphrase_ngram_range=(1,2), top_n=5)]))
    return {"keywords": keyphrases}
async def extract_graph_data(text):
    documents = [Document(page_content=text)]
    graph_documents = await graph_transformer.aconvert_to_graph_documents(documents)
    return graph_documents

def visualize_graph(graph_documents, output_file):
    net = Network(height="1000px", width="100%", directed=True,
                  notebook=False, bgcolor="#222222", font_color="white", filter_menu=True, cdn_resources='remote')

    nodes = graph_documents[0].nodes
    relationships = graph_documents[0].relationships

    node_dict = {node.id: node for node in nodes}
    valid_edges = []
    valid_node_ids = set()

    for rel in relationships:
        if rel.source.id in node_dict and rel.target.id in node_dict:
            valid_edges.append(rel)
            valid_node_ids.update([rel.source.id, rel.target.id])

    for node_id in valid_node_ids:
        node = node_dict[node_id]
        try:
            net.add_node(node.id, label=node.id, title=node.type, group=node.type)
        except:
            continue

    for rel in valid_edges:
        try:
            net.add_edge(rel.source.id, rel.target.id, label=rel.type.lower())
        except:
            continue

    net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -100,
                    "centralGravity": 0.01,
                    "springLength": 200,
                    "springConstant": 0.08
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based"
            }
        }
    """)

    net.save_graph(output_file)
    return output_file
@app.post("/graph/")
async def graph(data: TextInput):
    #code here
    graph_documents = await extract_graph_data(data.text)
    output_file = "knowledge_graph.html"
    visualize_graph(graph_documents, output_file)
    return FileResponse(output_file, media_type='text/html', filename="knowledge_graph.html")
if __name__ == "__main__":
    uvicorn.run("main", host="127.0.0.1", port=8000, reload=True)



