#!/usr/bin/env python3

import argparse
import json
import string
# from InvertedIndex import InvertedIndex
from nltk.stem import PorterStemmer

from InvertedIndex import InvertedIndex

def main() -> None:
    stemmer = PorterStemmer()
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build")
    tf_parser = subparsers.add_parser("tf")
    tf_parser.add_argument("doc_id", type=int)
    tf_parser.add_argument("term", type=str)

    
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    # args.command = "search"
    # args.query = "brave"
    # args.command = "tf"
    # args.doc_id = 424
    # args.term = 'trapper'
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
        case _:
            parser.print_help()


def get_stop_words():
    with open("./data/stopwords.txt") as text:
        return text.read().splitlines()
        
        

if __name__ == "__main__":
    main()