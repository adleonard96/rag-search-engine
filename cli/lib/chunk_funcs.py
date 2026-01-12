import re


def chunk(text: str, size: int, overlap: int):
    sections = text.split(" ")
    
    left = 0
    right = size
    
    res = []
    
    if right > len(sections):
        res.append(" ".join(sections))
    
    while right <= len(sections):
        res.append(" ".join(sections[left:right]))
        
        if right == len(sections):
            break
        left += size - overlap
        right += size
        if right > len(sections):
            right = len(sections)
    
    return res

def semantic_chunck(text: str, size: int, overlap: int):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        i += size - overlap
    return chunks