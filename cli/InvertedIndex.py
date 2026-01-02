from collections import defaultdict
import pickle
import os
import json
import string
from nltk.stem import PorterStemmer
from collections import Counter

class InvertedIndex:
    
    def __init__(self):
        self.index = defaultdict(list)
        self.docmap = defaultdict(str)
        self.term_frequencies = defaultdict(Counter)
        
    def __add_document(self, doc_id, text: str):
        self.docmap[doc_id] = text
        self.term_frequencies[doc_id].update(tokenize_text(text))
        for val in set(tokenize_text(text)):
            self.index[val].append(doc_id)
            
    def get_documents(self, term: str):
        return sorted(list(set(self.index[term.lower()])))

    def get_titles(self, doc_id):
        return self.docmap[doc_id]
    
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
        
        term_frequencies = open("./cache/term_frequencies.pkl", "wb")
        pickle.dump(self.term_frequencies, term_frequencies)
        term_frequencies.close()
    
    def load(self):
        if not os.path.isfile("./cache/index.pkl") or not os.path.isfile("./cache/docmap.pkl"):
            raise Exception
        with open("./cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("./cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("./cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)
            
    def get_tf(self, doc_id, term):
        if len(term.split()) > 1:
            raise Exception
        return self.term_frequencies[doc_id][term] if self.term_frequencies[doc_id][term] else 0
            
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = get_stop_words()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words



def get_stop_words():
    with open("./data/stopwords.txt") as text:
        return text.read().splitlines()
        
        