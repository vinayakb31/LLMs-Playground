def chunker(text: str, chunk_size: int, overlap: int) -> list:
    if overlap >= chunk_size:
        print("Overlap cannot be bigger than chunk size")
        return None
    
    chunks = list()
    left, right = 0, chunk_size-1
    
    while left < len(text):
        chunks.append(text[left:right+1])
        left += chunk_size-overlap
        right += chunk_size-overlap
    
    return chunks

print(chunker(text="ABCDEFGHIJKLMNOPQRSTUVWXYZ1234",chunk_size=10, overlap=3))