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
                
            current_chunk = [sentence]
            current_length = sentence_length
                    
        print(current_chunk)
                    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks

print(chunker(text =
    "This is a short sentence. "
    "This is an extremely long sentence that deliberately contains "
    "far more characters than our chosen chunk size so that we can "
    "test whether the chunker keeps the entire sentence intact. "
    "This is another short sentence.",
    chunk_size = 50, overlap=1
))