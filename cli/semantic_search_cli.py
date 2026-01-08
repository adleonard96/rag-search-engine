#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify")
    subparsers.add_parser("verify_embeddings")
    embed_parser = subparsers.add_parser("embed_text")
    embed_parser.add_argument("text")
    args = parser.parse_args()
    
    # args.command = "verify_embeddings"
    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()