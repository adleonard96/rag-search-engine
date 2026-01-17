import argparse
import json
from lib.semantic_search import read_movies
from hybrid_search import HybridSearch


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    data_set = None
    with open("./data/golden_dataset.json") as f:
        data_set = json.load(f)
    search = HybridSearch(read_movies())
    for case in data_set["test_cases"]:
        retrieved = search.rrf_search(case["query"], 60, limit)
        retrieved = list(map(lambda x: x[1]["title"], retrieved))
        match = set(retrieved) & set(case['relevant_docs'])
        
        recall_score = (len(match)/ len(set(case['relevant_docs'])))
        precision_score = (len(match) / len(retrieved))
        f1 = 2 * (precision_score * recall_score) / (precision_score + recall_score)
        print(f"Query: {case["query"]}")
        print(f"Precision@{limit}: {precision_score:.4f}")
        print(f"Recall@{limit}: {recall_score:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f'Retrieved: {", ".join(retrieved)}')
        print(f'Relevant: {", ".join(match)}')
if __name__ == "__main__":
    main()