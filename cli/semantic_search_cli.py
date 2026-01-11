#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search, chunk, semantic_chunck

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify")
    subparsers.add_parser("verify_embeddings")
    embed_parser = subparsers.add_parser("embed_text")
    embed_parser.add_argument("text")
    embedquery_parser = subparsers.add_parser("embedquery")
    embedquery_parser.add_argument("query")
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, nargs="?", default=5)
    chunk_parser = subparsers.add_parser("chunk")
    chunk_parser.add_argument("text")
    chunk_parser.add_argument("--chunk-size", type=int, nargs="?", default=200)
    chunk_parser.add_argument("--overlap", type=int, nargs="?", default=0)
    
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk")
    semantic_chunk_parser.add_argument("text")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs="?", default=4)
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs="?", default=0)
    
    args = parser.parse_args()
    # args.command = "verify_embeddings"
    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunks = chunk(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")    
            for i, chunk_val in enumerate(chunks):
                print(f"{i + 1}. {chunk_val}")
        case "semantic_chunk":
            chunks = semantic_chunck(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")    
            for i, chunk_val in enumerate(chunks):
                print(f"{i + 1}. {chunk_val}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()