import chromadb
import os
import json
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="my_collection")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

with open("ground_truth.json", "r") as f:
    ground_truth = json.load(f)

def search(query: str, n_results=10):
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

def rewrite_query(query: str, client=groq_client):
    system_prompt = """
    Rewrite the user's question to improve its suitability for
    document retrieval.

    Preserve the original meaning and intent.
    Use relevant technical terminology where appropriate.
    Do not introduce assumptions or information that is not
    implied by the original question.

    Return only the rewritten query, without explanations.
    """
    
    response = client.chat.completions.create(
        model = "openai/gpt-oss-20b",
        temperature = 0.2,
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )
    
    return response.choices[0].message.content

def multi_query_retrieval(query: str, k=60, client=groq_client):
    system_prompt = """
    You are a query generation assistant designed to improve
    document retrieval in a Retrieval-Augmented Generation (RAG) system.

    Generate exactly three alternative search queries based on
    the user's original question.

    Instructions:
    1. Preserve the original question's meaning and intent.
    2. Explore three different retrieval perspectives or aspects
    of the original question.
    3. Use relevant technical terminology, synonyms, and alternative
    expressions where appropriate.
    4. Each query should target a distinct perspective rather than
    simply rephrasing the same question.
    5. Do not introduce assumptions, unrelated concepts, or information
    that is not implied by the original question.
    6. Keep each query concise and suitable for semantic search.

    Return only the three queries, one per line.
    Do not include numbering, explanations, or additional text.
    """
    
    response = client.chat.completions.create(
        model = "openai/gpt-oss-20b",
        temperature = 0.2,
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )
    
    response = response.choices[0].message.content
    response = response.split('\n')
    response = [i.strip() for i in response]
    
    combined_results = {
        "ids": [],
        "documents": [],
        "metadatas": [],
        "distances": []
    }
    
    rankings = dict()
    
    for query in response:
        rank = 1
        search_results = search(query=query, n_results=10)
        
        for id, document, metadata, distance in zip(search_results['ids'][0], search_results['documents'][0],
                                                    search_results['metadatas'][0], search_results['distances'][0]):
            if id not in rankings:
                rankings[id] = [rank]
            else:
                rankings[id].append(rank)
                
            rank += 1
            
    for id, ranks in rankings.items():
        rrf_score = 0
        
        for i in ranks:
            rrf_score += 1/(k+i)
        
        rankings[id] = round(rrf_score, 5)
    
    return rankings

def generate_answer(prompt: str, client=groq_client):
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

def rag_pipeline(query: str, client=groq_client):
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
            if result['recall'][k] is not None:
                avg_recall[k].append(result['recall'][k])
            if result['precision'][k] is not None:
                avg_precision[k].append(result['precision'][k])

    n = len(eval_results)
    for k in k_values:
        avg_recall[k] = round(sum(avg_recall[k])/n, 2)
        avg_precision[k] = round(sum(avg_precision[k])/n, 2)

    return {"average_recall": avg_recall,"average_precision": avg_precision}        

def evaluate_retrieval(query, relevant_chunks, k_values):
    retrieved_chunks = search(query, n_results=max(k_values))['ids'][0]
    results = {}

    for k in k_values:
        results[k] = {
            "recall": recall_at_k(
                retrieved_chunks, relevant_chunks, k
            ),
            "precision": precision_at_k(
                retrieved_chunks, relevant_chunks, k
            )
        }

    return results

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
queries_to_test = [
    "What is the purpose of the TurboQuant project?",
    "Can I use TurboQuant without an internet connection?",
    "Does TurboQuant save previous conversations?"
]

test_ground_truth = [
    ["turboquant_0", "turboquant_1", "turboquant_2", "turboquant_3"],
    ["turboquant_0", "turboquant_1", "turboquant_2", "turboquant_3"],
    ["turboquant_4", "turboquant_5"]
]

for i in range(len(queries_to_test)):
    rewritten_query = rewrite_query(query=queries_to_test[i], client=groq_client)
    relevant_chunks = test_ground_truth[i]
    
    # results_normal = evaluate_retrieval(query=queries_to_test[i], relevant_chunks=relevant_chunks, k_values=k_values)
    # results_modified = evaluate_retrieval(query=rewritten_query, relevant_chunks=relevant_chunks, k_values=k_values)
    
    # results_normal = search(query=queries_to_test[i], n_results=10)
    # results_modified = search(query=rewritten_query, n_results=10)
    # results_expanded = search(query=expanded_query, n_results=10)
    
    # print(f"Query: {queries_to_test[i]}")
    # print(results_normal['ids'][0])
    # print(results_normal['distances'][0])
    # print('\n\n')
    
    # print(f"Query (Rewritten): {rewritten_query}")
    # print(results_modified['ids'][0])
    # print(results_modified['distances'][0])
    # print('\n\n')
    
    # print(f"Query (Expanded): {expanded_query}")
    # print(results_expanded['ids'][0])
    # print(results_expanded['distances'][0])
    # print('\n\n')
    
multi_query_retrieval(query=queries_to_test[0])