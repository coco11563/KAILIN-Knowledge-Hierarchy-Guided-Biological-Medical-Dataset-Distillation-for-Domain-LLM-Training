import json
import argparse
from tqdm.auto import tqdm

def filter_scores(input_file_path, output_file_path, score_threshold):
    """
    Filters out records from a JSONL file where llama_score or bio_score exceed the score_threshold.

    Args:
        input_file_path (str): Path to the input JSONL file.
        output_file_path (str): Path to the output JSONL file.
        score_threshold (float): The score threshold for filtering.
    """
    with open(input_file_path, "r", encoding="utf-8") as infile, open(output_file_path, "w", encoding="utf-8") as outfile:
        for line in tqdm(infile.readlines(), desc="Processing records"):
            data = json.loads(line)
            llama_score = data.get("llama_score", 0)
            bio_score = data.get("bio_score", 0)
            
            if llama_score <= score_threshold and bio_score <= score_threshold:
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter JSONL records based on score threshold.")
    parser.add_argument('--input_file_path', type=str, required=True, help='Path to the input JSONL file.')
    parser.add_argument('--output_file_path', type=str, required=True, help='Path to the output JSONL file.')
    parser.add_argument('--score_threshold', type=float, required=True, help='Score threshold for filtering records.')

    args = parser.parse_args()

    filter_scores(args.input_file_path, args.output_file_path, args.score_threshold)
