import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("my_collection")

data = collection.get(
    include=["documents", "metadatas"]
)

for i in data["ids"]:
    print(i)
    
for i in data["metadatas"]:
    print(i)