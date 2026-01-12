import json
import os
import numpy as np

from .chunk_funcs import (chunk, semantic_chunck)
from .semantic_search import SemanticSearch

EMBEDDINGS_LOCATION = "./cache/chunk_embeddings.npy"
META_DATA_LOCATION = "./cache/chunk_metadata.json"

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.documents = None
        self.document_map = {}
        
    def build_chunk_embeddings(self, documents):
        self.documents = documents
        chunks = []
        chunk_meta = []
        for i, document in enumerate(documents):
            self.document_map[document["id"]] = document
            text = document.get("description", "")
            if text.strip():
                SIZE = 4
                OVERLAP = 1
                curr_chuncks = semantic_chunck(document["description"], SIZE, OVERLAP)
                chunks += curr_chuncks
                for j, _ in enumerate(curr_chuncks):
                    chunk_meta.append({
                        "movie_idx": i,
                        "chunk_idx": j,
                        "total_chunks": len(chunks)
                    }) 

        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        np.save(EMBEDDINGS_LOCATION, self.embedding)
        with open(META_DATA_LOCATION, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        if os.path.exists(EMBEDDINGS_LOCATION) and os.path.exists(META_DATA_LOCATION):
            self.embedding = np.load('./cache/chunk_embeddings.npy')
            
            with open(META_DATA_LOCATION, 'r') as file:
                self.chunk_metadata = json.load(file)
        else:
            self.build_chunk_embeddings(documents)
        
        return self.chunk_embeddings