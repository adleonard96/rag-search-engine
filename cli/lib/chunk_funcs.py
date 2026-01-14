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
    text = text.strip()
    if len(text) == 0:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]
    # if len(sentences) == 1:
        
    chunks = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + size]
        
        for i in range(len(chunk_sentences)):
            chunk_sentences[i] = chunk_sentences[i].strip()
        chunk_sentences = list(filter(lambda x: len(x) > 0, chunk_sentences))
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        i += size - overlap
    return chunks