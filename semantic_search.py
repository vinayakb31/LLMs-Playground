import chromadb
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
client = chromadb.PersistentClient("./chroma_db/")

f1 = open("Sample Data/self_healing_waf.txt")
f2 = open("Sample Data/turboquant.txt")

sentences = []

for i in f1:
    sentences.append(i)
    
for i in f2:
    sentences.append(i)

embeddings = model.encode(sentences)
embedded_data = list()

for sentence, embedding in zip(sentences, embeddings):
    embedded_data.append({
        "text":sentence,
        "embedding":embedding.tolist()
    })
    
ids = []
for i in range(len(sentences)):
    ids.append(str(i))
    
collection = client.get_or_create_collection(name="my_collection")
existing = collection.get(ids=ids)

new_ids = [i for i in ids if i not in existing['ids']]
new_sentences, new_embeddings = [], []

for i in new_ids:
    new_sentences.append(embedded_data[int(i)]["text"])
    new_embeddings.append(embedded_data[int(i)]["embedding"])    
    
if len(new_ids) > 0:
    collection.add(documents=new_sentences, ids=new_ids, embeddings=new_embeddings)

query_embeddings = model.encode([
    'What is a WAF?',
    'What is quantisation?'
])

results = collection.query(
    query_embeddings=query_embeddings.tolist(),
    n_results=5
)

print(results['ids'])
print(results['documents'])
print(results['distances'])