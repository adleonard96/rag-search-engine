import time
from InvertedIndex import InvertedIndex

idx = InvertedIndex()
idx.load()
t0 = time.time()
idx.bm25_search("bear dicaprio", 5)
t1 = time.time()
print("search:", t1 - t0)