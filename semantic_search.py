from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

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
    
print(embedded_data[0])