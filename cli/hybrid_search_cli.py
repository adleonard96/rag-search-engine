import argparse
from collections import defaultdict
import json
import time
from lib.semantic_search import read_movies
from hybrid_search import HybridSearch, normalize
from query_enhance import (spell_check, rewrite, expand, rerank, batch)
from sentence_transformers import CrossEncoder

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
    rrf_search.add_argument("--rerank-method", nargs="?", choices=["individual", "batch", "cross_encoder"], type=str )
    
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
                print(f"{movie[1]["description"]}")
        case "rrf-search":
            if args.rerank_method:
                match args.rerank_method:
                    case "individual":
                        args.limit = args.limit * 5
                    case "batch":
                        args.limit = args.limit * 5
                    case "cross_encoder":
                        args.limit = args.limit * 5
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
            new_res = []
            exit = 0
            if args.rerank_method:
                match args.rerank_method:
                    case "individual":
                        for result in res:
                            result = list(result)
                            broke = True
                            failed = 0
                            while broke:
                                try:
                                    result[1]["rerank"] = rerank(args.query, result[1]["title"], result[1]["description"]).text.replace("\n", "")
                                    broke = False
                                except:
                                    failed += 1
                                    print(f"it failed {failed} times")    
                                time.sleep(3)
                            if result[1]["rerank"].isnumeric():
                                result[1]["rerank"] = int(result[1]["rerank"])
                            new_res.append(result)
                                    
                        res = new_res
                        res = sorted(res, key= lambda x: x[1]["rerank"], reverse=True)[:int(args.limit / 5)]            
                    case "batch":
                        exit = 1
                        llm_res = batch(args.query, res).text
                        llm_res = json.loads(llm_res.replace("\n", "").replace("`","").replace("json",""))
                        res_dict = defaultdict(dict)
                        for result in res:
                            res_dict[result[0]] = result[1]
                            
                        for i, res in enumerate(llm_res):
                            res_dict[res]["rerank"] = i + 1
                            
                        new_order = sorted(res_dict.items(), key=lambda x: x[1]["rerank"])[:int(args.limit/5)]
                        
                        for i, movie in enumerate(new_order):
                            print(f"{i + 1}. {movie[1]["title"]}")
                            print(f"Rerank Rank: {movie[1]["rerank"]}/10")
                            print(f"RRF: {movie[1]["rrf_score"]:.3f}")
                            print(f"BM25 Rank: {movie[1]['bm25_rank']:.3f}, Semantic Rank: {movie[1]["semantic_rank"]:.3f}")
                            print(f"{movie[1]["description"]}")
                    case "cross_encoder":
                        pairs = []
                        for doc in res:
                            pairs.append([args.query, f"{doc[1].get('title', '')} - {doc[1].get('document', '')}"])
            if exit == 0:     
                for i, movie in enumerate(res):
                    print(f"{i + 1}. {movie[1]["title"]}")
                    if movie[1]["rerank"]:
                        print(f"Rerank Score: {movie[1]["rerank"]}/10")
                    print(f"RRF: {movie[1]["rrf_score"]:.3f}")
                    print(f"BM25 Rank: {movie[1]['bm25_rank']:.3f}, Semantic Rank: {movie[1]["semantic_rank"]:.3f}")
                    print(f"{movie[1]["description"]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()