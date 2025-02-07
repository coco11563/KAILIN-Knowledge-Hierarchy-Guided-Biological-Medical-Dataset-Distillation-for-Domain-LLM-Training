from mteb import MTEB
from sentence_transformers import SentenceTransformer
import os
import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Embedding model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path or Name to the Embedding model.')
    parser.add_argument('--save_folder', default=f"results", type=str, required=True, help='Path to save the output.')
    parser.add_argument('--task', default="BIOSSES", type=str, required=True, help='type of the benchmark need, like BIOSSES.')

    args = parser.parse_args()

    model_path = args.model_path
    save_folder = args.save_folder
    task = args.task

    model = SentenceTransformer(model_path, trust_remote_code=True)
    evaluation = MTEB(tasks=[task])
    results = evaluation.run(model, output_folder=f"{save_folder}/{task}")
  