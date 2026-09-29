from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

sentences = ['There are seven days in a week.',
            'Honeybees represent cleanliness.',
            'I love machine learning.']

embeddings = model.encode(sentences)

for s, e in zip(sentences, embeddings):
    print("Sentence:", s)
    print("Embedding:", e)