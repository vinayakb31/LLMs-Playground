import json

with open("ground_truth.json", "r") as f:
    truth = json.load(f)

queries = [
    "What is a web application firewall?",
    "How does a WAF compare to a traditional firewall?",
    "Who developed the turboquant algorithm?",
    "What is 4-bit quantisation?",
    "Can an RTX 3050 run 4-bit quantisation?",
    "What does risk_engine.py do?",
    "What GPT does turboquant recommend?",
    "How much faster is turboquant compared to traditional quantisation algorithms?",
    "What is the capital of France?",
    "What is photosynthesis?",
    "Who is the president of France?"
]

for query in queries:
    # results = search(query=query)
    
    for i in truth:
        if i['query'] == query:
            rel_chunks = i['relevant_chunks']
            break

    print(rel_chunks)