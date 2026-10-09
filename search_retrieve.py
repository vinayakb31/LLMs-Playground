import chromadb
import os
import json
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq
from IPython.display import display, Markdown

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_collection")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def search(query: str, n_results=3):
    query_embeddings = model.encode(query)

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results
    )
    
    results['query'] = query
    return results

def filtering(threshold: float, results):
    filtered_results = []
    
    for dist, doc, src in zip(results['distances'][0], results['documents'][0], results['metadatas'][0]):
        if dist <= threshold:
            filtered_results.append({
                "query": results['query'],
                "distance": dist,
                "source": src,
                "document": doc
            })
        
    return filtered_results

def build_context(filtered_results):
    return '\n'.join(i['document'] for i in filtered_results)

def build_prompt(query: str, context: str):
    instruction = "Here is some information retrieved from the knowledge base. Use it to answer the user's question."
    prompt = f"Instruction:\n{instruction}\n\nContext:\n{context}\n\nQuestion:{query}"

    return prompt

def generate_answer(prompt: str, client):
    response = client.chat.completions.create(
        model = "openai/gpt-oss-20b",
        temperature = 0.2,
        messages = [
            {
                "role": "system",
                "content": prompt
            }
        ]
    )
    
    return response.choices[0].message.content

def rag_pipeline(query: str, client):
    results = search(query)
    filtered_results = filtering(threshold=0.8, results=results)
    
    if not filtered_results:
        return None
    
    context = build_context(filtered_results=filtered_results)
    prompt = build_prompt(query=query, context=context)
    answer = generate_answer(prompt=prompt, client=client)
    
    return answer

def recall_at_k(retrieved_chunks: list, relevant_chunks: list, k: int) -> float | None:
    if not relevant_chunks:
        return None
    
    recall_value = 0
    
    for chunk in relevant_chunks:
        if chunk in retrieved_chunks[:k]:
            recall_value += 1
            
    return round(recall_value/len(relevant_chunks), 2)

def precision_at_k(retrieved_chunks: list, relevant_chunks: list, k: int) -> float | None:
    precision_value = 0
    
    for chunk in retrieved_chunks[:k]:
        if chunk in relevant_chunks:
            precision_value += 1
        
    return round(precision_value/k, 2)

queries = [
    "What is a web application firewall?",
    "How does a WAF compare to a traditional firewall?",
    "Who developed the turboquant algorithm?",
    "What is 4-bit quantisation?",
    "Can an RTX 3050 run 4-bit quantisation?",
    "What does risk_engine.py do?",
    "What GPT does turboquant recommend?",
    "How much faster is turboquant compared to traditional quantisation algorithms?",
    "What is the capital of France?",
    "What is photosynthesis?",
    "Who is the president of France?"
]
        
results = search(query=queries[0], n_results=10)
rel_chunks = ["self_healing_waf_1", "self_healing_waf_2"]

k_values = [1, 3, 5, 10]
for k in k_values:
    recall_value = recall_at_k(retrieved_chunks=results['ids'][0],
                               relevant_chunks=rel_chunks,
                               k=k)
    
    precision_value = precision_at_k(retrieved_chunks=results['ids'][0],
                               relevant_chunks=rel_chunks,
                               k=k)
    
    print(f"k={k}\tRecall@k={recall_value}\tPrecision@k={precision_value}")