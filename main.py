# from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File

from fastapi.responses import FileResponse
import torch
from pydantic import BaseModel
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
from keybert import KeyBERT
import google.generativeai as genai


from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from pyvis.network import Network
# import asyncio
# import uvicorn
import os
from sentence_transformers import SentenceTransformer, util
load_dotenv()
app = FastAPI()

# Models
api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key)
graph_transformer = LLMGraphTransformer(llm=llm)
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# qa_model = pipeline('question-answering')
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
    input_text = "summarize: " + data.text.strip().replace("\n", " ")
    
    # Tokenize
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    # Generate summary
    summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return {"summary": summary}

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




