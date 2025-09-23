import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.embeddings import embed_query, embed_passage
import torch


def main():
    print("PyTorch device available:", torch.cuda.is_available())

    example_query = "What is the capital of France?"
    example_passage = "Paris is the capital and most populous city of France."

    print("Embedding query:", example_query)
    query_vec = embed_query(example_query)
    print("Query embedding length:", len(query_vec))
    print("Query embedding first 5 values:", query_vec[:5])

    print("Embedding passage:", example_passage)
    passage_vec = embed_passage(example_passage)
    print("Passage embedding length:", len(passage_vec))
    print("Passage embedding first 5 values:", passage_vec[:5])


if __name__ == "__main__":
    main()