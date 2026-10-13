import chromadb
import os
import json
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_collection")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

with open("ground_truth.json", "r") as f:
    ground_truth = json.load(f)

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
            
    return recall_value/len(relevant_chunks)

def precision_at_k(retrieved_chunks: list, relevant_chunks: list, k: int) -> float | None:
    if not relevant_chunks:
        return None
    
    precision_value = 0
    
    for chunk in retrieved_chunks[:k]:
        if chunk in relevant_chunks:
            precision_value += 1

    if not retrieved_chunks[:k]:
        return None

    return precision_value / len(retrieved_chunks[:k])

def evaluation(queries: list, ground_truth: list, k_values: list):
    eval_results = []
    
    for query in queries:
        search_results = search(query=query, n_results=max(k_values))
        retrieved_chunks = search_results['ids'][0]

        for i in ground_truth:
            rel_chunks = []
            
            if i['query'] == query:
                rel_chunks = i['relevant_chunks']
                break

        if not rel_chunks:
            continue
            
        query_result = {
            "query": query,
            "recall": {},
            "precision": {},
            "relevant_retrieved": {},
            "relevant_missed": []
        }

        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if chunk in rel_chunks:
                query_result["relevant_retrieved"][chunk] = rank
                    
        for chunk in rel_chunks:
            if chunk not in retrieved_chunks:
                query_result["relevant_missed"].append(chunk)

        for k in k_values:
            query_result["recall"][k] = recall_at_k(
                retrieved_chunks, rel_chunks, k
            )

            query_result["precision"][k] = precision_at_k(
                retrieved_chunks, rel_chunks, k
            )

        eval_results.append(query_result)
        
    return eval_results

def calculate_average(eval_results: list, k_values: list):
    avg_recall, avg_precision = dict(), dict()
    for k in k_values:
        avg_recall[k], avg_precision[k] = [], []

    for result in eval_results:
        for k in k_values:
            avg_recall[k].append(result['recall'][k])
            avg_precision[k].append(result['precision'][k])

    n = len(eval_results)
    for k in k_values:
        avg_recall[k] = round(sum(avg_recall[k])/n, 2)
        avg_precision[k] = round(sum(avg_precision[k])/n, 2)

    return {"average_recall": avg_recall,"average_precision": avg_precision}        
    

queries = [
    # --- Self-Healing WAF: answerable ---
    "What is a web application firewall?",
    "What accuracy does the self-healing WAF achieve?",
    "Which datasets are used to train or evaluate the WAF?",
    "What role does OWASP Juice Shop play in the WAF project?",
    "How does the WAF defend against evasion attacks?",
    "How does the Hybrid Risk Engine make decisions?",
    "What is the inference latency of the WAF?",
    "Which Python file implements the hybrid risk engine?",
    "Which scripts are used in the WAF data augmentation pipeline?",
    "How does adversarial augmentation help the WAF?",

    # --- TurboQuant: answerable ---
    "What is the purpose of the TurboQuant project?",
    "How does TurboQuant support private, offline inference?",
    "What hardware does TurboQuant target?",
    "What software prerequisites are required to run TurboQuant?",
    "Does TurboQuant support persistent conversation history?",

    # --- Questions the corpus cannot answer ---
    "Who developed the TurboQuant algorithm?",
    "What GPT model does TurboQuant recommend?",
    "How much faster is TurboQuant than traditional quantisation?",
    "Can an RTX 3050 run TurboQuant reliably?",
    "What is the capital of France?",
]

k_values = [1, 3, 5, 10]
eval_results = evaluation(queries=queries, ground_truth=ground_truth, k_values=k_values)
search_results = search(query=queries[0], n_results=10)

for rank, id in enumerate(search_results['ids'][0], start=1):
    print(f"{rank}\t{id}")