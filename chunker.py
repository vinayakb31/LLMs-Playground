import re
import emoji
import json
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-minilm-l6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_collection")

with open("Sample Data/self_healing_waf.txt", encoding="utf-8") as f1:
    file1 = f1.read()
    
with open("Sample Data/turboquant.txt", encoding="utf-8") as f2:
    file2 = f2.read()

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
        documents.append(chunk["text"])
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

embedded_chunks = list(generator(files))
addToChroma(embedded_chunks=embedded_chunks)