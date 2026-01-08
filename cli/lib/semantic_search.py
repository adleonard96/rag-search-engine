import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 
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

def verify_embeddings():
    movies = None
    with open("./data/movies.json", 'r') as file:
        data = json.load(file)
        movies = data.get("movies")
    s = SemanticSearch()
    embeddings = s.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")