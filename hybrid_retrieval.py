import re
import chromadb
import math
from sentence_transformers import SentenceTransformer

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="my_collection")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def search(query: str, n_results: int = 10) -> list[dict]:
    """Run dense vector search in ChromaDB and return ranked chunk records."""
    query_embeddings = model.encode(query)

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    ranked_results = []
    for rank, (chunk_id, doc, meta, dist) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        ranked_results.append({
            "chunk_id": chunk_id,
            "document": doc,
            "rank": rank,
            "distance": round(dist, 5),
            "metadata": meta,
        })

    return ranked_results

def tokenise(text: str) -> list:    
    tokens = re.split(r'[\s,!?;:()]+', text.lower())

    for i in range(len(tokens)):
        while tokens[i] and tokens[i][-1] in ['.',',','?','!']:
            tokens[i] = tokens[i][:-1]
        
    tokens = [token for token in tokens if token.strip()]
    
    return tokens

def calculate_document_frequency(tokens_all: list):
    document_frequency = {}
    
    for tokens in tokens_all:
        for token in set(tokens):
            if token in document_frequency.keys():
                document_frequency[token] += 1
            else:
                document_frequency[token] = 1
                
    return document_frequency

def calculate_term_frequency(tokens_all: list):
    term_frequency = []
    
    for tokens in tokens_all:
        frequency = {}
        for token in tokens:
            if token in frequency.keys():
                frequency[token] += 1
            else:
                frequency[token] = 1
            
        term_frequency.append(frequency)
        
    return term_frequency

def calculate_inverse_document_frequency(document_frequency: dict, tokens_all: list):
    idf = {}
    n = len(tokens_all)
    
    for token in document_frequency.keys():
        idf[token] = round(math.log(1 + ((n-document_frequency[token]+0.5)/(document_frequency[token]+0.5))), 5)
                
    return idf

def calculate_bm25(
    chunk_ids: list,
    documents: list,
    metadatas: list,
    query: str,
    tokens_all: list,
    tf: list,
    idf: dict,
) -> list[dict]:
    """Score all corpus documents with BM25 and return descending ranked chunk records."""
    query_tokens = tokenise(text=query)

    b = 0.75
    k1 = 1.5
    avgdl = sum(len(tokens) for tokens in tokens_all) / len(tokens_all)
    lnf = [1 - b + b * len(tokens) / avgdl for tokens in tokens_all]

    scored = []
    for i in range(len(tf)):
        bm25 = 0.0
        for token in query_tokens:
            if token in tf[i]:
                bm25 += idf[token] * (tf[i][token] * (k1 + 1) / (tf[i][token] + k1 * lnf[i]))

        scored.append({
            "chunk_id": chunk_ids[i],
            "document": documents[i],
            "score": round(bm25, 5),
            "metadata": metadatas[i],
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    for rank, item in enumerate(scored, start=1):
        item["rank"] = rank

    return scored


chroma_documents = collection.get(include=["documents", "metadatas"])
chunk_ids = chroma_documents["ids"]
chunks = chroma_documents["documents"]
metadatas = chroma_documents["metadatas"]

tokens_all = [tokenise(text=document) for document in chunks]
tf = calculate_term_frequency(tokens_all=tokens_all)
df = calculate_document_frequency(tokens_all=tokens_all)
idf = calculate_inverse_document_frequency(document_frequency=df, tokens_all=tokens_all)

if __name__ == "__main__":
    query = "hybrid risk engine"

    bm25_results = calculate_bm25(
        chunk_ids=chunk_ids,
        documents=chunks,
        metadatas=metadatas,
        query=query,
        tokens_all=tokens_all,
        tf=tf,
        idf=idf,
    )
    vector_results = search(query=query, n_results=10)

    print(f"Corpus size: {len(chunks)} chunks")
    print("\nTop 5 BM25 results:")
    for r in bm25_results[:5]:
        print(f"  Rank {r['rank']} | {r['chunk_id']} | score={r['score']}")

    print("\nTop 5 Vector results:")
    for r in vector_results[:5]:
        print(f"  Rank {r['rank']} | {r['chunk_id']} | distance={r['distance']}")