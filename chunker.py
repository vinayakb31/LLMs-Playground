import re

def chunker(text: str, chunk_size: int, overlap: int) -> list:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = list()
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        candidate_length = sentence_length + current_length
        
        if current_chunk:
            candidate_length += 1
            
        if candidate_length <= chunk_size:
            current_chunk.append(sentence)
            current_length = candidate_length
            
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            if overlap > 0: 
                current_chunk = current_chunk[-overlap:]+[sentence]
            else:
                current_chunk = [sentence]
            
            current_length = sum(len(i) for i in current_chunk) + len(current_chunk) - 1
                                        
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks

print(chunker(text =
    '''Python is a programming language.
    It is popular for AI.
    It is also widely used in Data Science.
    Python has a large ecosystem.
    Many libraries are available.''',
    chunk_size = 70, overlap=0
))