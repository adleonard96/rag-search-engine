import argparse
from lib.semantic_search import read_movies
from hybrid_search import HybridSearch, normalize

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    normailze_parser = subparser.add_parser("normalize")
    normailze_parser.add_argument("nums", nargs="+", type=float)
    
    weighted_search = subparser.add_parser("weighted-search")
    weighted_search.add_argument("query")
    weighted_search.add_argument("--alpha", nargs="?", type=float, default=0.5)
    weighted_search.add_argument("--limit", nargs="?", type=int, default=5)
    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            res = normalize(args.nums)
            for num in res:
                print(f"* {num:.4f}")
        case "weighted-search":
            hybrid = HybridSearch(read_movies())
            res = hybrid.weighted_search(args.query, args.alpha, args.limit)
            
            for i, movie in enumerate(res):
                print(f"{i + 1}. {movie[1]["title"]}")
                print(f"Hybrid Score: {movie[1]["hybrid_score"]:.3f}")
                print(f"BM25: {movie[1]['bm25_score']:.3f}, Semantic: {movie[1]["semantic_score"]:.3f}")
                print(f"{movie[1]["desciption"]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()