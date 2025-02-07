import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from itertools import islice
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from tqdm.auto import tqdm
# from sim_mesh.sim_mesh_plus import  compare_multiple_lists





# RAQG_embedding = HuggingFaceEmbeddings(model_name=RAQG_emb_model,model_kwargs={'device': 'cuda:7'})

def init_embeddings(emb_model_path, cuda_device=4):
    # emb_model = "/home/cxx/work/FlagEmbedding/sft/step2_0403/checkpoint-8500"
    embeddings = HuggingFaceEmbeddings(model_name=emb_model_path, model_kwargs={'device': f'cuda:{cuda_device}'})
    return embeddings

# Pubmed_embedding = HuggingFaceEmbeddings(model_name=Pubmed_emb_model,model_kwargs={'device': 'cuda:7'})
# BGE_embedding = HuggingFaceEmbeddings(model_name=BGE_emb_model,model_kwargs={'device': 'cuda:7'})


def Qdrant_multi(emb_name,collection_name):
    return Qdrant(
        client=QdrantClient(url="http://10.0.82.180:6333", timeout=1000),    #
        collection_name=collection_name, 
        embeddings=emb_name,
    )

TOP_K = 8


def process_list(input_list):
    new_list = []
    for sublist in input_list:
        processed_sublist = []
        for item in sublist:
            processed_item = item.split('/')[0].strip()
            processed_sublist.append(processed_item)
        new_list.append(processed_sublist)
    return new_list

def get_mesh(emb_docs):
    mesh_list = []
    for _ in emb_docs:
        mesh_list.append(_.metadata["mesh"])
    new_list = process_list(mesh_list)
    return new_list

def build_mesh_tree_dict(mesh_disc_file):
    mesh_name_dict = dict()
    # mesh_tree_dict = dict()
    with open(mesh_disc_file, 'r') as f:
        for line in (f.readlines()):
            data = json.loads(line)
            ui = data['DescriptorUI']
            # tree_list = data['TreeNumberList']
            name = data['DescriptorName']
            # mesh_tree_dict[ui] = tree_list
            mesh_name_dict[name] = ui
    return mesh_name_dict

def query_in_out(raw_sft_data, embedding_model_path, mesh_name_dict, query_key1, query_key2, TOP_K):

    embeddings = init_embeddings(embedding_model_path)

    result_ = []
    
    for _ in tqdm(raw_sft_data):
        
        raw_sft_data_dict = json.loads(_)
        raw_pmid= raw_sft_data_dict["pmid"]
        raw_mesh = raw_sft_data_dict["raw_mesh"]
        raw_title = raw_sft_data_dict['title']
        query_1 = raw_sft_data_dict[query_key1]
        query_2 = raw_sft_data_dict[query_key2]


        raw_mesh_ui = []
        for mesh in raw_mesh:
            if mesh not in mesh_name_dict.keys():
                # print(mesh)
                continue
            raw_mesh_ui.append(mesh_name_dict[mesh])
        
        docs_1 = Qdrant_multi(embeddings,"pubmed_2024").similarity_search(query=query_1,k=TOP_K)
        docs_2 = Qdrant_multi(embeddings,"pubmed_2024").similarity_search(query=query_2,k=TOP_K)
        
        mesh_1 = get_mesh(docs_1)
        mesh_2 = get_mesh(docs_2)

        mesh_ui_1 = []
        for mesh_list_temp in mesh_1:
            tempp = []
            for mesh in mesh_list_temp:
                if mesh not in mesh_name_dict.keys():
                    continue
                tempp.append(mesh_name_dict[mesh])
            mesh_ui_1.append(tempp)
        
        mesh_ui_2 = []
        for mesh_list_temp in mesh_2:
            tempp = []
            for mesh in mesh_list_temp:
                if mesh not in mesh_name_dict.keys():
                    continue
                tempp.append(mesh_name_dict[mesh])
            mesh_ui_2.append(tempp)
        

        out = {'pmid':raw_pmid, 
               'title':raw_title,
               'raw_context':raw_sft_data_dict['context'],
               'raw_mesh':raw_mesh_ui, 
               f'{query_key1}':query_1, 
               f'{query_key1}_recall_mesh':mesh_ui_1, 
               f'{query_key2}':query_2,
               f'{query_key2}_recall_mesh':mesh_ui_2
            }
        result_.append(out)

    return result_

def read_chunks(file_path, chunk_size=1000):
    with open(file_path, 'r') as file:
        while True:
            chunk = list(islice(file, chunk_size))
            if not chunk:
                break
            yield chunk



def main(file_path, 
         out_file,
         embedding_model_path,
         mesh_disc_file,
         top_k,
         query_key1,
         query_key2,
         cuda_device=4,
         num_processes=10):
    
    mesh_name_dict = build_mesh_tree_dict(mesh_disc_file)
    multiprocessing.set_start_method('spawn')
    # num_processes = 8
    # chunk_size = 100
    
    
    writer = open(out_file, 'w')
    # with open(raw_sft_query_with_mesh_jsonl_path, "r", encoding="utf-8") as file:
    #     lines = file.readlines()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for chunk in read_chunks(file_path, 1000):
            futures.append(executor.submit(query_in_out, chunk, embedding_model_path, mesh_name_dict, query_key1, query_key2, top_k))
        
        results = [future.result() for future in futures]

        # print(results[0][0])
    
    
        # print(f"Processed {sum(results)} documents in total.")
    for list in results:
        for item in list:
            # print(type(item))
            writer.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retrieval the relevant Docs.")
    parser.add_argument('--input_file', type=str, required=True, help='Path of file with the two or more querys.')
    parser.add_argument('--output_file', type=str, required=True, help='Path of the ouput file.')
    parser.add_argument('--mesh_desc_file', type=str, required=True, help='like the /home/cxx/work/Q_Generation/mesh/mesh_decs2.jsonl, to parse the mesh tree.')
    parser.add_argument('--model_path', type=str, required=True, help='Path or Name to the Embedding model.')
    parser.add_argument('--query_key1', type=str, required=True, help='Key param to get the query1')
    parser.add_argument('--query_key2', type=str, required=True, help='Key param to get the query2')
    parser.add_argument('--Top_k', default="4", type=int, required=True, help='num of retrieval docs.')

    args = parser.parse_args()

    embedding_model_path = args.model_path
    mesh_desc_file = args.mesh_desc_file
    input_file = args.input_file
    out_file = args.output_file
    top_k = args.Top_k
    query_key1 = args.query_key1
    query_key2 = args.query_key2
    main(input_file, out_file, embedding_model_path, mesh_desc_file, top_k, query_key1, query_key2)
    








