import re

def tokenise(text: str) -> list:    
    tokens = re.split(r'[\s,!?;:()]+', text.lower())

    for i in range(len(tokens)):
        while tokens[i] and tokens[i][-1] in ['.',',','?','!']:
            tokens[i] = tokens[i][:-1]
        
    tokens = [token for token in tokens if token.strip()]
    
    return tokens

print(tokenise("The risk_engine.py is a file to analyse risks."))
print(tokenise("Hello... world!!!"))
print(tokenise("CSE-CIC-IDS2018 uses augment_data.py."))
print(tokenise("..."))