#!/usr/bin/env python3

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    # args.command = "search"
    # args.query = "Great"
    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            with open("./data/movies.json", 'r') as file:
                data = json.load(file)
                movies = data.get("movies")
                res = []
                for movie in movies:
                    if args.query in movie['title']: 
                        res.append(movie["title"])
                
                [print(f"{x + 1}. {res[x]}") for x in range(5)] 
                
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()