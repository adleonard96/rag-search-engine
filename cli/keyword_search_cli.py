#!/usr/bin/env python3

import argparse
import json
import math
import string
# from InvertedIndex import InvertedIndex
from nltk.stem import PorterStemmer

from InvertedIndex import InvertedIndex
from constants import Constants

def main() -> None:
    stemmer = PorterStemmer()
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build")
    tfidf_parser = subparsers.add_parser("tfidf")
    tfidf_parser.add_argument("doc_id", type=int)
    tfidf_parser.add_argument("term")
    
    idf_parser = subparsers.add_parser("idf")
    idf_parser.add_argument("term")
    
    tf_parser = subparsers.add_parser("tf")
    tf_parser.add_argument("doc_id", type=int)
    tf_parser.add_argument("term", type=str)

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=Constants.BM25_K1.value, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=Constants.BM25_B.value, help="Tunable BM25 b parameter")
    
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", nargs="?", type=int, default=5, help="Limit results")
    
    args = parser.parse_args()
    # args.command = "bm25search"
    # args.query = "space adventure"
    # args.limit = 5
    
    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            stop_words = get_stop_words()
            res = []
            query_args = list(filter(lambda x: x not in stop_words, args.query.lower().translate(str.maketrans("", "", string.punctuation)).split()))
            query_args = list(map(lambda x: stemmer.stem(x), query_args))
            index = InvertedIndex()
            index.load()
            for arg in query_args:
                res += index.get_documents(arg)
                if len(res) >= 5:
                    break
            movie_list_count = len(res) if len(res) <= 5 else 5
            [print(f"{res[x]} {index.get_titles(res[x])}") for x in range(movie_list_count)] 
        case "build":
            builder = InvertedIndex()
            builder.build()
            builder.save()
        case "tf":
            counter = InvertedIndex()
            counter.load()
            print(f'{counter.get_tf(int(args.doc_id), args.term)}')
        case "idf":
            data = InvertedIndex()
            print(f"Inverse document frequency of '{args.term}': {data.get_idf(args.term):.2f}")
        case "tfidf":
            data = InvertedIndex()
            tf_idf = data.get_tf_idf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            data = InvertedIndex()
            bm25idf = data.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":   
            data = InvertedIndex()
            data.load()
            bm25tf = data.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            data = InvertedIndex()
            data.load()
            res = data.bm25_search(args.query, args.limit)
            for i in range(len(res)):
                doc_id = res[i].key()
                score = res[i].values()[0]
                print(f"{i + 1} ({doc_id}) {data.get_titles(doc_id)} - Score: {score:.2f}")
        case _:
            parser.print_help()


def get_stop_words():
    with open("./data/stopwords.txt") as text:
        return text.read().splitlines()
        
        

if __name__ == "__main__":
    main()