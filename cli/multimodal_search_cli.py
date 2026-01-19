import argparse
from collections import defaultdict
from lib.multimodal_search import verify_image_embedding, MultimodalSearch
from lib.semantic_search import read_movies

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparser.add_parser("verify_image_embedding")
    verify_parser.add_argument("path")
    
    image_search = subparser.add_parser("image_search")
    image_search.add_argument("path")
    
    
    args = parser.parse_args()
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.path)
        case "image_search":
            movie_list = []
            mapping = defaultdict(dict)
            for movie in read_movies():
                movie_list.append(f"{movie["title"]}: {movie["description"]}")
                mapping[f"{movie["title"]}: {movie["description"]}"] = {
                    "id": movie["id"],
                    "title": movie["title"],
                    "description": movie["description"]
                }
            search = MultimodalSearch(movie_list, mapping=mapping)
            search.search_with_image(args.path)
            
        case _:
            parser.print_help()
            

if __name__ == "__main__":
    main()