import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_collection")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

query = str(input("Query: "))
query_embeddings = model.encode(query)

results = collection.query(
    query_embeddings=query_embeddings,
    n_results=3
)

print(results)