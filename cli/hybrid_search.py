from collections import defaultdict
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
        mapper = defaultdict(dict)
        
        nc_scores = normalize(list(map(lambda x: x["score"], chunk_res)))
        nb_scores = normalize(list(map(lambda x: x["score"], b25_res)))
        
        for i in range(len(nc_scores)):
            id = chunk_res[i]["id"]
            mapper[id]["title"] = self.idx.docmap[id]['title']
            mapper[id]["desciption"] = self.idx.docmap[id]['description'][:100]
            mapper[id]["semantic_score"] = nc_scores[i]
        
        for i in range(len(nb_scores)):
            id = b25_res[i]["id"]
            mapper[id]["bm25_score"] = nb_scores[i]
            
        for key in mapper:
            bm25 = mapper[key].get("bm25_score", 0)
            semantic_score = mapper[key].get("semantic_score", 0)
            mapper[key]["hybrid_score"] = self.hybrid_score(bm25, semantic_score, alpha)
            
        return sorted(mapper.items(), key=lambda x: x[1]["hybrid_score"], reverse= True)[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

    def hybrid_score(self, bm25_score, semantic_score, alpha=0.5):
        return alpha * bm25_score + (1 - alpha) * semantic_score
    
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
        