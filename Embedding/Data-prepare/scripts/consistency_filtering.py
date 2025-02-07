import argparse
import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

def get_all_files_in_folder(folder_path):
    """
    获取指定文件夹下的所有文件路径

    Param:
    folder_path (str): 文件夹的路径

    Return:
    list: 包含所有文件路径的列表，如果路径不存在或不是文件夹，则返回空列表
    """
    file_list = []

    if not os.path.exists(folder_path):
        print(f"Error: The folder path '{folder_path}' does not exist.")
        return file_list

    if not os.path.isdir(folder_path):
        print(f"Error: The path '{folder_path}' is not a directory.")
        return file_list

    try:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                file_list.append(file_path)
            else:
                print(f"Warning: {file_path} is not a file.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return file_list

def ensure_str(data):
    """
    确保输入数据为字符串，如果是列表则合并成一个字符串

    Param:
    data: 输入数据，可以是字符串或列表

    Return:
    str: 合并后的字符串
    """
    # if isinstance(data, list):
    #     return ' '.join(data)
    # return data
    return ' '.join(data) if isinstance(data, list) else data

def get_data(file_list, out_file_path, model, sim_threshold, key_x, key_y):
    """
    处理文件列表中的数据，根据相似度筛选并保存结果

    Param:
    file_list (list): 包含所有文件路径的列表
    out_file_path (str): 输出文件的路径
    model: 用于计算句子嵌入和相似度的模型实例
    sim_threshold (float): 相似度阈值
    key_x (str): 用于获取摘要的键
    key_y (str): 用于获取标题的键

    Return:
    None
    """
    try:
        with open(out_file_path, 'w', encoding='utf-8') as writer:
            for file_name in tqdm(file_list, desc="Processing files"):
                if not os.path.isfile(file_name):
                    print(f"Warning: {file_name} is not a file or does not exist.")
                    continue

                with open(file_name, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc=f"Reading lines from {file_name}"):
                        try:
                            data = json.loads(line)
                            abstract = ensure_str(data.get(key_x, []))
                            title = ensure_str(data.get(key_y, ""))

                            if not abstract or not title:
                                continue

                            sentences = [abstract, title]
                            embeddings = model.encode(sentences)
                            sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

                            if sim < sim_threshold:
                                continue

                            outdata = {
                                'query': title,
                                'document': abstract,
                                'cos_sim': sim,
                                'metadata': data.get('Metadata', {})
                            }

                            writer.write(json.dumps(outdata, ensure_ascii=False) + '\n')

                        except json.JSONDecodeError:
                            print(f"Error: Failed to decode JSON line in file {file_name}.")
                        except Exception as e:
                            print(f"Error processing line in file {file_name}: {e}")
    except Exception as e:
        print(f"Error writing to output file {out_file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and filter data based on similarity.")
    parser.add_argument('--model_path', type=str, required=True, help='Path or Name to the Embedding model')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the input folder containing files')
    parser.add_argument('--out_file_path', type=str, required=True, help='Path to the output file')
    parser.add_argument('--sim_threshold', type=float, required=True, help='Similarity threshold for filtering, > threshold would be saved')
    parser.add_argument('--key_x', type=str, required=True, help='Key for sentence_1 in the JSON data')
    parser.add_argument('--key_y', type=str, required=True, help='Key for sentence_2 in the JSON data')

    args = parser.parse_args()

    model_path = args.model_path
    folder_path = args.folder_path
    out_file_path = args.out_file_path
    sim_threshold = args.sim_threshold
    key_x = args.key_x
    key_y = args.key_y

    model = SentenceTransformer(model_path)
    file_list = get_all_files_in_folder(folder_path)
    get_data(file_list, out_file_path, model, sim_threshold, key_x, key_y)

