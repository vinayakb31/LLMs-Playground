import chromadb

client = chromadb.PersistentClient("./chroma_db/")

collection = client.get_or_create_collection("my_collection")
ids = ['0','1','2','3']
existing = collection.get(ids)
print(existing)
print(existing.keys())

new_ids = [i for i in ids if i not in existing['ids']]
