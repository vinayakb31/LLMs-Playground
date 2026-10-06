import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_collection")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def search(query: str, n_results=3):
    query_embeddings = model.encode(query)

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results
    )
    
    return results

def filtering(threshold: float, results):
    filtered_results = []
    
    for dist, doc in zip(results['distances'][0], results['documents'][0]):
        if dist <= threshold:
            filtered_results.append({
                "distance": dist,
                "document": doc
            })
        
    return filtered_results

queries = ['What is the capital of France?',
           'What is a WAF?',
           'Does 4-bit quantisation run on an RTX 3050?']

for query in queries:
    results = search(query)
    filtered_results = filtering(threshold=0.8, results=results)

    if not filtered_results:
        print("Lack of Context")

# embeddingA = model.encode("MacOS is really nice to use.")
# embeddingB = model.encode("Windows feels so trash.")

# magnitude = np.linalg.norm(embeddingA)
# print("Magnitude of Embedding A:", magnitude)

# l2_distance = np.linalg.norm(embeddingA-embeddingB)
# print("L2 Distance:", l2_distance)

# l2_squared = np.sum((embeddingB-embeddingA)**2)
# print("L2 Squared:", l2_squared)

# cosine_similarity = np.dot(embeddingA, embeddingB) / (
#     np.linalg.norm(embeddingA) * np.linalg.norm(embeddingB)
# )
# print("Cosine Similarity:", cosine_similarity)