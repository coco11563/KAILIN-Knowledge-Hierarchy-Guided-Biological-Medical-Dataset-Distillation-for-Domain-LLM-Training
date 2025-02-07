import os
import json
import logging
import sqlite3
import math
import sys
import argparse
from tqdm.auto import tqdm
import numpy as np


def load_database_to_memory(disk_db_path):
    """
    Loads an SQLite database from disk into memory.

    Args:
        disk_db_path (str): Path to the SQLite database file on disk.

    Returns:
        sqlite3.Connection: A connection to the in-memory database.
    """
    memory_conn = sqlite3.connect(':memory:')
    disk_conn = sqlite3.connect(disk_db_path)
    disk_conn.backup(memory_conn)
    disk_conn.close()
    return memory_conn

def load_frequency_data_from_jsonl(jsonl_path):
    """
    Loads frequency data from a JSONL file into a dictionary.

    Args:
        jsonl_path (str): Path to the JSONL file.

    Returns:
        dict: A dictionary containing the frequency data.
    """
    category_freq_dicts = {}
    with open(jsonl_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            category_freq_dicts.update(data)
    return category_freq_dicts

def query_db(query, args=(), one=False):
    """
    Executes a SQL query on the in-memory database.

    Args:
        query (str): The SQL query to execute.
        args (tuple): The arguments for the SQL query.
        one (bool): Whether to fetch only one result.

    Returns:
        list or tuple: The result of the query.
    """
    cur = memory_conn.cursor()
    try:
        cur.execute(query, args)
        rv = cur.fetchall()
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        rv = []
    finally:
        cur.close()
    return (rv[0] if rv else None) if one else rv

def getFreqDict(mesh_id):
    """
    Retrieves frequency data for a given MeSH ID from the database and frequency dictionary.

    Args:
        mesh_id (str): The MeSH ID.

    Returns:
        dict: A dictionary of category frequency data.
    """
    mesh_data = query_db(
        "SELECT TreeNumberList FROM DescriptorData WHERE DescriptorUI = ?",
        [mesh_id],
        one=True,
    )[0].split(";")
    
    categories = set(item.split('.')[0] for item in mesh_data)
    category_list = list(categories)

    if not mesh_data:
        sys.stderr.write(f"MeSH ID not found in the database :::: {mesh_id}\n")
        return None

    category_freq_dicts = {}
    for category in category_list:
        if category in category_frequency_data:
            category_freq_dicts[category] = category_frequency_data[category]
        else:
            sys.stderr.write(f"No frequency data found for category :::: {category}\n")
            category_freq_dicts[category] = None  
    return category_freq_dicts

def get_all_descendants(mesh_id):
    """
    Retrieves all descendants of a given MeSH ID from the database.

    Args:
        mesh_id (str): The MeSH ID.

    Returns:
        list: A list of tuples containing the descendant MeSH IDs and their categories.
    """
    descendants = []
    stack = [mesh_id]
    while stack:
        current_id = stack.pop()
        children = query_db("SELECT * FROM ChildrenData WHERE Parent = ?", [current_id])
        for child in children:
            child_id, category = child[0], child[4]
            if (child_id, category) not in descendants:
                descendants.append((child_id, category))
                stack.append(child_id)
    return descendants

def getMeSHIC(mesh_id):
    """
    Calculates the Information Content (IC) for a given MeSH ID.

    Args:
        mesh_id (str): The MeSH ID.

    Returns:
        dict: A dictionary of IC values for each category.
    """
    freqDict = getFreqDict(mesh_id)
    if freqDict is None:
        sys.stderr.write(f"词频统计错误 {mesh_id} \n")
        return None
    children = get_all_descendants(mesh_id)
    IC_by_category = {}
    for category, freqs in freqDict.items():
        freqTotal = sum(
            freqs.get(child, 0)
            for child, child_cat in children
            if child_cat == category
        )
        IC = -math.log(freqTotal, 10) if 0 < freqTotal <= 1 else 0
        IC_by_category[category] = IC
    return IC_by_category

def getMaxIC(mesh_id):
    """
    Retrieves the maximum IC for each category for a given MeSH ID.

    Args:
        mesh_id (str): The MeSH ID.

    Returns:
        dict: A dictionary of maximum IC values for each category.
    """
    mesh_data = query_db(
        "SELECT * FROM FrequencyData WHERE MeSHID = ?",
        [mesh_id],
        one=True, 
    )
    categories = mesh_data[2].split(";")
    MaxIC_dict = {}
    for category in categories:
        rows = query_db(
            "SELECT * FROM MaxICData WHERE Category = ?",
            [category],
            one=True,
        )
        MaxIC_dict[category] = rows[1]
    return MaxIC_dict

def check_shared_categories(mesh1, mesh2, categories):
    """
    Checks if two MeSH IDs share any categories.

    Args:
        mesh1 (str): The first MeSH ID.
        mesh2 (str): The second MeSH ID.
        categories (dict): A dictionary of categories.

    Returns:
        bool: True if they share any categories, False otherwise.
    """
    return bool(categories[mesh1] & categories[mesh2])

def find_lowest_common_ancestor(trees1, trees2):
    """
    Finds the lowest common ancestor (LCA) for two sets of tree numbers.

    Args:
        trees1 (list): The first set of tree numbers.
        trees2 (list): The second set of tree numbers.

    Returns:
        str: The tree number of the LCA.
    """
    lca = None
    for t1 in trees1:
        path1 = t1.split(".")
        for t2 in trees2:
            path2 = t2.split(".")
            common_path = []
            for p1, p2 in zip(path1, path2):
                if p1 == p2:
                    common_path.append(p1)
                else:
                    break
            common_path_str = ".".join(common_path)
            if lca is None or len(common_path_str) > len(lca):
                lca = common_path_str
    return lca

def lca_tree(mesh_id_1, mesh_id_2):
    """
    Finds the lowest common ancestor (LCA) of two MeSH IDs.

    Args:
        mesh_id_1 (str): The first MeSH ID.
        mesh_id_2 (str): The second MeSH ID.

    Returns:
        str: The LCA MeSH ID, or None if not found.
    """
    try:
        tree_numbers_1 = query_db(
            "SELECT TreeNumberList FROM DescriptorData WHERE DescriptorUI = ?",
            [mesh_id_1],
            one=True,
        )[0].split(";")
        tree_numbers_2 = query_db(
            "SELECT TreeNumberList FROM DescriptorData WHERE DescriptorUI = ?",
            [mesh_id_2],
            one=True,
        )[0].split(";")
        lca = find_lowest_common_ancestor(tree_numbers_1, tree_numbers_2)

        if lca:
            lca_mesh_id = query_db(
                "SELECT DescriptorUI FROM DescriptorTreeData WHERE TreeNumber = ?",
                [lca],
                one=True,
            )
            return lca_mesh_id[0] if lca_mesh_id else None
        return None

    except Exception as e:
        logging.error(f"Error in lca_tree function: {str(e)}")
        return None

def calculate_ictmps(lca_IC, MaxIC_1, MaxIC_2):
    """
    Calculates the IC-TMP values for the given categories.

    Args:
        lca_IC (dict): IC values for the lowest common ancestor.
        MaxIC_1 (dict): Maximum IC values for the first MeSH ID.
        MaxIC_2 (dict): Maximum IC values for the second MeSH ID.

    Returns:
        dict: A dictionary of IC-TMP values for each category.
    """
    ictmps = {}
    for category in lca_IC:
        if (
            category in MaxIC_1
            and category in MaxIC_2
            and MaxIC_1[category] > 0
            and MaxIC_2[category] > 0
        ):
            ictmp_1 = lca_IC[category] / MaxIC_1[category]
            ictmp_2 = lca_IC[category] / MaxIC_2[category]
            ictmps[category] = (ictmp_1 + ictmp_2) / 2
    return ictmps

def calculate_metrics(ictmps, summed_ic, MaxIC_1, MaxIC_2):
    """
    Calculates the Lin, Jiang, and REL metrics.

    Args:
        ictmps (dict): IC-TMP values.
        summed_ic (dict): Summed IC values.
        MaxIC_1 (dict): Maximum IC values for the first MeSH ID.
        MaxIC_2 (dict): Maximum IC values for the second MeSH ID.

    Returns:
        tuple: Dictionaries of Lin, Jiang, and REL values for each category.
    """
    lin = {}
    jiang = {}
    rel = {}

    for category in ictmps:
        if category in summed_ic and summed_ic[category] > 0:
            max_ic = max(MaxIC_1.get(category, 0), MaxIC_2.get(category, 0))
            average_ictmp = ictmps[category]

            lin[category] = 2 * average_ictmp / summed_ic[category]
            jiang[category] = 1 - min(1, summed_ic[category] - 2 * average_ictmp)
            rel[category] = lin[category] * (1 - (10 ** (-average_ictmp * max_ic)))

    return lin, jiang, rel

def simple_average(results):
    """
    Calculates the simple average of a list of dictionaries.

    Args:
        results (list): A list of dictionaries with numeric values.

    Returns:
        float: The overall average value.
    """
    average_results = {}
    for key in results[0]:
        average_results[key] = sum(result[key] for result in results) / len(results)
    overall_average = sum(average_results.values()) / len(average_results)
    return overall_average

def simil(mesh_id_1, mesh_id_2):
    """
    Calculates the similarity between two MeSH IDs.

    Args:
        mesh_id_1 (str): The first MeSH ID.
        mesh_id_2 (str): The second MeSH ID.

    Returns:
        float: The similarity score between 0 and 1.
    """
    if mesh_id_1 == mesh_id_2:
        return 1

    MaxIC_1 = getMaxIC(mesh_id_1)
    MaxIC_2 = getMaxIC(mesh_id_2)

    IC_1 = getMeSHIC(mesh_id_1)
    IC_2 = getMeSHIC(mesh_id_2)

    summed_ic = {
        key: IC_1.get(key, 0) + IC_2.get(key, 0) for key in set(IC_1).union(IC_2)
    }

    lca = lca_tree(mesh_id_1, mesh_id_2)
    if lca:
        lca_IC = getMeSHIC(lca)
        if lca_IC:
            ictmps = calculate_ictmps(lca_IC, MaxIC_1, MaxIC_2)
            lin, jiang, rel = calculate_metrics(ictmps, summed_ic, MaxIC_1, MaxIC_2)
            scores = [v for dct in (lin, jiang, rel) for v in dct.values()]
            if scores:
                normalized_scores = [max(0, min(1, score)) for score in scores]
                return sum(normalized_scores) / len(normalized_scores)
    return 0

def compare_lists(list1, list2):
    """
    Compares two lists of MeSH IDs and calculates the average similarity score.

    Args:
        list1 (list): The first list of MeSH IDs.
        list2 (list): The second list of MeSH IDs.

    Returns:
        float: The average similarity score between the lists.
    """
    scores = []
    for mesh_id_1 in list1:
        for mesh_id_2 in list2:
            score = simil(mesh_id_1, mesh_id_2)
            scores.append(score)
    return np.mean(scores)

def compare_multiple_lists(list1, list_of_lists):
    """
    Compares one list of MeSH IDs to multiple lists and calculates the overall average similarity score.

    Args:
        list1 (list): The list of MeSH IDs to compare.
        list_of_lists (list): A list of lists of MeSH IDs.

    Returns:
        float: The overall average similarity score.
    """
    overall_scores = []
    for sublist in list_of_lists:
        score = compare_lists(list1, sublist)
        overall_scores.append(score)
    return np.mean(overall_scores)

def process_jsonl_and_compute_averages_preference_dataset(jsonl_file_path, output_file_path, mesh_desc_file):
    """
    Processes a JSONL file to compute and compare similarity scores for llama and bio recall data.

    Args:
        jsonl_file_path (str): Path to the input JSONL file.
        output_file_path (str): Path to the output JSONL file.
        mesh_desc_file (str): Path to the mesh description JSONL file.

    Returns:
        None
    """
    mesh_name_id_dict = {}
    with open(mesh_desc_file, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            mesh_name_id_dict[data["DescriptorName"]] = data["DescriptorUI"]

    with open(jsonl_file_path, "r", encoding="utf-8") as file:
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            for line in tqdm(file.readlines()):
                try:
                    data = json.loads(line)
                    pubmed_id = data["id"]["pubmed"]

                    raw_mesh_pre = data["mesh"]
                    raw_mesh = []
                    for temp in raw_mesh_pre:
                        temp = temp['UI']
                        if temp in mesh_name_id_dict.values():
                            raw_mesh.append(temp)

                    llama_recall = data['llama_recall']
                    llama_mesh_list = []
                    for recall in llama_recall:
                        mesh_id_list_1_pre = recall['metadata']['mesh']
                        mesh_id_list_1 = []
                        for mesh in mesh_id_list_1_pre:
                            mesh = mesh.split('/')[0].strip()
                            if mesh in mesh_name_id_dict.keys():
                                mesh_id_list_1.append(mesh_name_id_dict[mesh])
                        llama_mesh_list.append(mesh_id_list_1)
                    llama_score = compare_multiple_lists(raw_mesh, llama_mesh_list)

                    bio_recall = data['bio_recall']
                    bio_mesh_list = []
                    for recall in bio_recall:
                        mesh_id_list_1_pre = recall['metadata']['mesh']
                        mesh_id_list_1 = []
                        for mesh in mesh_id_list_1_pre:
                            mesh = mesh.split('/')[0].strip()
                            if mesh in mesh_name_id_dict.keys():
                                mesh_id_list_1.append(mesh_name_id_dict[mesh])
                        bio_mesh_list.append(mesh_id_list_1)
                    bio_score = compare_multiple_lists(raw_mesh, bio_mesh_list)

                    if bio_score is not None and llama_score is not None:
                        result = {
                            'id': pubmed_id,
                            'title': data['title'],
                            'raw_context': data['raw_context'],
                            'llama_q': data['llama_q'],
                            'llama_score': llama_score,
                            'bio_q': data['bio_q'],
                            'bio_score': bio_score
                        }
                        outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                        logging.info(
                            f"Processed and written result for PubMed ID {pubmed_id}"
                        )
                    else:
                        logging.info(
                            f"Error when writing result for PubMed ID {pubmed_id}: score is 0!"
                        )
                except Exception as e:
                    logging.error(f"Error processing PubMed ID {pubmed_id}: {e}")
                    continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute similarity scores for MeSH data.")
    parser.add_argument('--jsonl_file_path', type=str, required=True, help='Path to the input JSONL file.')
    parser.add_argument('--output_file_path', type=str, required=True, help='Path to the output JSONL file.')
    parser.add_argument('--mesh_desc_file', type=str, required=True, help='Path to the mesh description JSONL file.')
    parser.add_argument('--database_path', type=str, required=True, help='Path to the SQLite database file.')
    parser.add_argument('--frequency_jsonl_path', type=str, required=True, help='Path to the frequency JSONL file.')

    args = parser.parse_args()

    # Initialize global variables
    memory_conn = load_database_to_memory(args.database_path)
    category_frequency_data = load_frequency_data_from_jsonl(args.frequency_jsonl_path)

    # Process the JSONL file and compute scores
    process_jsonl_and_compute_averages_preference_dataset(
        args.jsonl_file_path,
        args.output_file_path,
        args.mesh_desc_file
    )
