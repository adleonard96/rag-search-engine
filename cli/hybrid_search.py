import os

from InvertedIndex import InvertedIndex
from lib.chunked_semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        b25_res = self._bm25_search(query, limit=limit * 500)
        chunk_res = self.semantic_search.search_chunks(query, limit=limit * 500)

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize(vals: list[float]):
    if len(vals) == 0:
        return
    min_val = min(*vals)
    max_val = max(*vals)
    second_half = max_val - min_val
    if min_val == max_val:
        return [1] * len(vals)
    
    res = []
    
    for val in vals:
        res.append((val - min_val)/second_half)
    
    return res
        