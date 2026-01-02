#!/usr/bin/env python3

import argparse
import json
import string


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    args.command = "search"
    args.query = "furious fast"
    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            with open("./data/movies.json", 'r') as file:
                data = json.load(file)
                movies = data.get("movies")
                res = []
                query_args = args.query.lower().translate(str.maketrans("", "", string.punctuation)).split()
                for movie in movies:
                    title = movie['title'].lower().translate(str.maketrans("", "", string.punctuation))
                    for arg in query_args:
                        if arg in title:
                            res.append(movie['title'])
                            break
                
                movie_list_count = len(res) if len(res) <= 5 else 5
                [print(f"{x + 1}. {res[x]}") for x in range(movie_list_count)] 
                
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()