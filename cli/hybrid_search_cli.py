import argparse
from lib.semantic_search import read_movies
from hybrid_search import HybridSearch, normalize
from query_enhance import (spell_check, rewrite, expand)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    normailze_parser = subparser.add_parser("normalize")
    normailze_parser.add_argument("nums", nargs="+", type=float)
    
    weighted_search = subparser.add_parser("weighted-search")
    weighted_search.add_argument("query")
    weighted_search.add_argument("--alpha", nargs="?", type=float, default=0.5)
    weighted_search.add_argument("--limit", nargs="?", type=int, default=5)
    
    rrf_search = subparser.add_parser("rrf-search")
    rrf_search.add_argument("query")
    rrf_search.add_argument("-k", nargs="?", type=int, default=60)
    rrf_search.add_argument("--limit", nargs="?", type=int, default=5)
    rrf_search.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    
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
        case "rrf-search":
            if args.enhance:
                match args.enhance:
                    case "spell":
                        llm_res = spell_check(args.query)
                        print( f"Enhanced query ({args.enhance}): '{args.query}' -> '{llm_res.text}'\n")
                        args.query = llm_res.text
                    case "rewrite":
                        llm_res = rewrite(args.query)
                        print( f"Enhanced query ({args.enhance}): '{args.query}' -> '{llm_res.text}'\n")
                        args.query = llm_res.text
                    case "expand":
                        llm_res = expand(args.query)
                        print( f"Enhanced query ({args.enhance}): '{args.query}' -> '{llm_res.text}'\n")
                        args.query = llm_res.text
                    case _:
                        pass
            hybrid = HybridSearch(read_movies())
            res = hybrid.rrf_search(args.query, args.k, args.limit)
            
            for i, movie in enumerate(res):
                print(f"{i + 1}. {movie[1]["title"]}")
                print(f"RRF: {movie[1]["rrf_score"]:.3f}")
                print(f"BM25 Rank: {movie[1]['bm25_rank']:.3f}, Semantic Rank: {movie[1]["semantic_rank"]:.3f}")
                print(f"{movie[1]["desciption"]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()