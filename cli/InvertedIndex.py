from collections import defaultdict
import math
import pickle
import os
import json
import string
from typing import Any
from nltk.stem import PorterStemmer
from collections import Counter
from constants import Constants

class InvertedIndex:
    
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = defaultdict(str)
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = defaultdict(int)
        self.index_path =  "./cache/index.pkl"
        
    def __add_document(self, doc_id, text: str):
        self.term_frequencies[doc_id].update(tokenize_text(text))
        self.doc_lengths[doc_id] = len(tokenize_text(text))
        for val in set(tokenize_text(text)):
            self.index[val].add(doc_id)
    
    def __get_avg_doc_length(self) -> float:
        doc_total = len(self.doc_lengths.keys())
        total = 0
        for doc in self.doc_lengths.values():
            total += doc
        
        return  total / doc_total
         
    def get_documents(self, term: str):
        return sorted(list(set(self.index[term.lower()])))

    def get_titles(self, doc_id):
        return self.docmap[doc_id]
    
    def build(self):
        with open("./data/movies.json", 'r') as file:
            data = json.load(file)
            movies = data.get("movies")
            
            for movie in movies:
                self.docmap[movie["id"]] = movie
                self.__add_document(movie["id"], f"{movie["title"]} {movie["description"]}")
    
    def get_doc_count(self):
        return len(self.docmap.keys())
    
    def get_idf(self, term):
        # self.load()
        total_docs = self.get_doc_count()
        total_matches = len(self.get_documents(tokenize_text(term)[0]))
        return math.log((total_docs + 1) / (total_matches + 1))
    
    def __get_document_frequencies(self, term):
        return len(self.get_documents(term))
    
    def get_bm25_tf(self, doc_id, term, k1=Constants.BM25_K1.value, b=Constants.BM25_B.value):
        term = tokenize_text(term)[0]
        length_norm = 1- b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        tf = self.get_tf(doc_id=doc_id, term=term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def bm25(self, doc_id, term):
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)
    
    def bm25_search(self, query, limit):
        tokens = tokenize_text(query)
        scores = defaultdict(float)
        for doc_id in self.docmap:
            score = 0.0
            for token in tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score
        
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc_id,
                title=doc["title"],
                document=doc,
                score=score,
            )
            results.append(formatted_result)

        return results
        
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
        
        lengths_file = open("./cache/doc_lengths.pkl", "wb")
        pickle.dump(self.doc_lengths, lengths_file)
        lengths_file.close()
    
    def load(self):
        if not os.path.isfile("./cache/index.pkl") or not os.path.isfile("./cache/docmap.pkl"):
            raise Exception
        with open("./cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("./cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("./cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open("./cache/doc_lengths.pkl", "rb") as f:
            self.doc_lengths = pickle.load(f)
            
    def get_tf(self, doc_id, term):
        if len(term.split()) > 1:
            raise Exception
        return self.term_frequencies[doc_id][term] if self.term_frequencies[doc_id][term] else 0
    
    def get_tf_idf(self, doc_id, term):
        # self.load()
        term = tokenize_text(term)[0]
        tf = self.get_tf(doc_id=doc_id, term=term)
        idf = self.get_idf(term=term)
        return tf * idf
    
    def get_bm25_idf(self, term: str) -> float:
        # self.load()
        term = tokenize_text(term)
        if len(term) != 1:
            raise Exception
        term = term[0]
        
        N = self.get_doc_count()
        df = len(self.index[term])
        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
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
        
SCORE_PRECISION = 3
def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }