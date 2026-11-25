import re
import math

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

def calculate_bm25(documents:list, query: str, tokens_all, tf, idf):
    query = tokenise(text=query)
    
    b = 0.75
    k1 = 1.5
    avgdl = 0
    lnf = []
    bm25_scores = []
    
    for tokens in tokens_all:
        avgdl += len(tokens)
        
    avgdl /= len(tokens_all)
    
    for i in range(len(tokens_all)):
        #calculating length-normalisation factor
        lnf_curr = 1 - b + b * len(tokens_all[i])/avgdl
        lnf.append(lnf_curr)
        
    for i in range(len(tf)):
        bm25 = 0
        
        for token in query:
            if token in tf[i].keys():
                bm25 += idf[token] * (tf[i][token]*(k1+1) / (tf[i][token]+k1*lnf[i]))
        
        bm25_scores.append([documents[i], round(bm25, 5)])
    
    bm25_scores = sorted(bm25_scores, key=lambda x: x[1], reverse=True)
    
    return bm25_scores

documents = [
    "Python is used to build a self healing WAF.",
    "The WAF uses Python for risk analysis and threat detection.",
    "The risk_engine.py file handles risk analysis.",
    "The augment_data.py script generates adversarial attack samples.",
    "Python supports machine learning and data processing.",
    "Adversarial attacks attempt to bypass WAF security.",
    "Python Python Python is used to build a WAF."
]

tokens_all = []
for document in documents:
    tokens_all.append(tokenise(text=document))
    
tf = calculate_term_frequency(tokens_all=tokens_all)
df = calculate_document_frequency(tokens_all=tokens_all)
idf = calculate_inverse_document_frequency(document_frequency=df, tokens_all=tokens_all)

query = "python"

print(calculate_bm25(documents=documents, query=query, tokens_all=tokens_all, tf=tf, idf=idf))

# print(tf)
# print("\n\n")
# print(idf)