from collections import defaultdict
import pickle
import os
import json

class InvertedIndex:
    
    def __init__(self):
        self.index = defaultdict(list)
        self.docmap = defaultdict(str)
        
    def __add_document(self, doc_id, text: str):
        self.docmap[doc_id] = text
        for val in text.split(" "):
            self.index[val.lower()].append(doc_id)
            
    def get_documents(self, term: str):
        return sorted(self.index[term.lower()])
    
    def build(self):
        with open("./data/movies.json", 'r') as file:
            data = json.load(file)
            movies = data.get("movies")
            
            for movie in movies:
                self.__add_document(movie["id"], f"{movie["title"]} {movie["description"]}")
                
    def save(self):
        os.makedirs("cache", exist_ok=True)
        
        index_file = open("./cache/index.pkl", "wb")
        pickle.dump(self.index, index_file)
        index_file.close()
        
        docmap_file = open("./cache/docmap.pkl", "wb")
        pickle.dump(self.docmap, docmap_file)
        docmap_file.close()
        