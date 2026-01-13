import json
import os
import re
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name) 
        self.embedding = []
        self.documents = None
        self.document_map = {}
                
    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        str_rep = []
        for document in documents:
            self.document_map[document["id"]] = document
            str_rep.append(f"{document["title"]}: {document["description"]}")
        
        self.embedding = self.model.encode(str_rep, show_progress_bar=True)
        np.save("./cache/movie_embeddings.npy", self.embedding)
        
        
        return self.embedding
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        str_rep = []
        for document in documents:
            self.document_map[document["id"]] = document
            str_rep.append(f"{document["title"]}: {document["description"]}")
        if os.path.exists('./cache/movie_embeddings.npy'):
            self.embedding = np.load('./cache/movie_embeddings.npy')
        if len(self.embedding) == len(documents):
            return self.embedding
        else:
            return self.build_embeddings(documents)
        
    def generate_embedding(self, text: str):
        if len(text.replace(" ", "")) == 0:
            raise ValueError
        return self.model.encode([text])[0]
    
    def search(self, query, limit):
        if len(self.embedding) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        embedding = self.generate_embedding(query)
        similarity: list[tuple] = []
        for i in range(len(self.embedding)):
            similarity.append((cosine_similarity(self.embedding[i], embedding), self.documents[i]))

        similarity.sort(key=lambda x: x[0], reverse=True)        
        
        results = []
        for score, doc in similarity[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results
        
def verify_model():
    model = SemanticSearch()
    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")
    
def embed_text(text):
    model = SemanticSearch()
    embedding = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    

def read_movies():
    with open("./data/movies.json", 'r') as file:
        data = json.load(file)
        return data.get("movies")
    
def verify_embeddings():
    movies = read_movies()
    s = SemanticSearch()
    embeddings = s.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    
def embed_query_text(query):
    s = SemanticSearch()
    embedding = s.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
    
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search(query, limit):
    s = SemanticSearch()
    s.load_or_create_embeddings(read_movies())
    results = s.search(query, limit)
    
    for i, res in enumerate(results):
        print(f"{i + 1}. {res["title"]} (score: {res["score"]})\n{res["description"]}")
        
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
    sections = re.split(r"(?<=[.!?])\s+", text)
    
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