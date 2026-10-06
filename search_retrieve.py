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

queries = ["What is a WAF?"]

filtered_results = []
for query in queries:
    results = search(query)
    filtered_results.extend(filtering(threshold=0.8, results=results))

# for i in filtered_results:
#     print(i, end='\n\n')
        
print(build_context(filtered_results=filtered_results))