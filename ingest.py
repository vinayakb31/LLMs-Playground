import re
import emoji
import json
import chromadb
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
ollama_client = OpenAI(
    base_url="http://192.168.1.14:11434/v1",
    api_key="ollama"
)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_collection")

DOCUMENTS_DIR = os.path.join("data", "documents")
ANALYSIS_DIR = os.path.join("data", "analysis")

with open(os.path.join(DOCUMENTS_DIR, "self_healing_waf.txt"), encoding="utf-8") as f1:
    file1 = f1.read()
    
with open(os.path.join(DOCUMENTS_DIR, "turboquant.txt"), encoding="utf-8") as f2:
    file2 = f2.read()
    
def load_prompt(prompt_name: str) -> str:
    prompt_path = os.path.join("prompts", prompt_name)
    
    with open(prompt_path, 'r') as f:
        return f.read()

def analyse_document(file, client=groq_client):
    system_prompt = load_prompt(prompt_name='document_analysis.txt')
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        max_completion_tokens=8192,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": file
            }
        ]
    )
    
    return response.choices[0].message.content

def generate_chunks(text: str, filename: str, chunk_size: int, overlap: int) -> list:
    sentences = emoji.replace_emoji(text, replace="")
    sentences = re.sub(r"\s*\n\s*", " ", sentences)
    sentences = re.split(r"(?<=[.!?])\s+", sentences)
    
    chunks = list()
    current_chunk = []
    current_length = 0
    idx = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        candidate_length = sentence_length + current_length
        
        if current_chunk:
            candidate_length += 1
            
        if candidate_length <= chunk_size:
            current_chunk.append(sentence)
            current_length = candidate_length
            
        else:
            if current_chunk:
                chunks.append({
                    "chunk_id" : str(filename) + "_" + str(idx),
                    "source": filename,
                    "text" : ' '.join(current_chunk)
                    })
                idx += 1
            
            if overlap > 0: 
                current_chunk = current_chunk[-overlap:]+[sentence]
            else:
                current_chunk = [sentence]
            
            current_length = sum(len(i) for i in current_chunk) + len(current_chunk) - 1
                                        
    if current_chunk:
        chunks.append({
                    "chunk_id" : str(filename) + "_" + str(idx),
                    "source": filename,
                    "text" : ' '.join(current_chunk)
                    })     
           
    return chunks

def contextualise_chunks(chunks, analysed_file, client=groq_client):    
    system_prompt = load_prompt(prompt_name='chunk_contextualise.txt')
    
    document_analysis_path = os.path.join(ANALYSIS_DIR, analysed_file)
    with open(document_analysis_path, "r") as f:
        document_analysis = json.load(f)
        
    contextualised_chunks = list()
    
    for i in range(len(chunks)):
        response = client.chat.completions.create(
            model="qwen2.5:14b",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"DOCUMENT ANALYSIS:\n{document_analysis}\n\nCHUNK TO CONTEXTUALISE:\n{chunks[i]['text']}"
                }
            ]
        )
        
        chunks[i]['contextualised_chunk'] = response.choices[0].message.content
    
    return chunks

def embed_chunks(model, chunks) -> list:    
    chunks_text = []
    embedded_chunks = chunks

    for i in embedded_chunks:
        chunks_text.append(i['text'])
        
    embeddings = model.encode(chunks_text)
    
    for i in range(len(embeddings)):
        embedded_chunks[i]['embedding'] = embeddings[i].tolist()
    
    return embedded_chunks
    
def generator(files: list):
    for file in files:
        chunks = generate_chunks(text=file["file"], filename=file["filename"], chunk_size=40, overlap=1)
        embedded_chunks = embed_chunks(model, chunks)
        
        for chunk in embedded_chunks:
            yield chunk

def addToChroma(embedded_chunks):
    ids, documents, embeddings, metadatas = [], [], [], []
    
    for chunk in embedded_chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["contextualised_chunk"])
        embeddings.append(chunk["embedding"])
        metadatas.append({"source": chunk["source"]})
        
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        
    return None

files = [
    {
        "file": file1,
        "filename": "self_healing_waf"
    },
    {
        "file": file2,
        "filename": "turboquant"
    }
]

# embedded_chunks = list(generator(files))
# addToChroma(embedded_chunks=embedded_chunks)

chunks = generate_chunks(text=file1, filename='waf', chunk_size=40, overlap=0)
# print(chunks)

chunks_contextualised = contextualise_chunks(chunks=chunks, analysed_file='self_healing_waf.json', client=ollama_client)
for chunk in chunks_contextualised:
    print(chunk,end='\n\n')