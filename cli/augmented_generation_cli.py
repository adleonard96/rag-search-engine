import argparse
from lib.semantic_search import read_movies
from hybrid_search import HybridSearch
from query_enhance import rrf_llm_response, summarize

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("query")
    summarize_parser.add_argument("--limit", nargs="?", type=int, default=5)
    
    
    args = parser.parse_args()

    search = HybridSearch(read_movies())
    
    match args.command:
        case "rag":
            query = args.query
            rrf_res = search.rrf_search(query, 60, 5)
            llm_res = rrf_llm_response(query, rrf_res)
            
            print("Search Results:")
            for _, result in enumerate(rrf_res):
                print(f"- {result[1]['title']}")
            
            print("RAG Response:")
            print(llm_res.text)
        case "summarize":
            query = args.query
            rrf_res = search.rrf_search(query, 60, args.limit)
            
            llm_res = summarize(query, rrf_res)
            
            print("Search Results:")
            for _, result in enumerate(rrf_res):
                print(f"- {result[1]['title']}")
            
            print("LLM Summary:")
            print(llm_res.text)
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()