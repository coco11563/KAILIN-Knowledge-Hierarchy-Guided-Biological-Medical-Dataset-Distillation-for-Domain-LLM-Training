## Prepare Data for DPO training and Do the DPO.

### Procedure of data prepare：

----
#### 1. Inference: Infer using Different model to get multiple Questions.



#### 2. Vector_DB Building: Build the vector database using Qdrant.
(including the PubMed abstract info, the metadata like the date, authors, especially the MeSH terms). 

#### 3. Docs Recalling: Recall N documents based on their similarity to Questions, and their corresponding MeSH.

Use a certain Embedding model, and two or more generated Questions as input, recall TOP_K docs, along with its relevant MeSH terms, in a Qdrant Database.

Before get the retrieval Docs, it's necessary to format a mesh-dict json file.

**Format MeSH file from XML file**: use the script: `mesh_process/process_xml.py`

to parse XML files to construct files in json format.

```
python mesh_process/process_xml.py \
	--xml_file /path/to/desc2024.xml \
	--out_file /path/to/mesh_desc.jsonl
```

**MeSH Sample**: use the scripts: use the script: `mesh_process/mesh_analysis.py`

to perform path-based node counting on each MeSH tree structure.

```
python mesh_process/mesh_analysis.py \
	--mesh_desc_path /path/to/mesh_desc.jsonl \
	--pubmed_path /path/to/pubmed_data.json \
	--tree_output_path /path/to/mesh_tree.json \
	--tree_sampled_output_path /path/to/mesh_tree_sampled.json

```

**Retrieval**: use the script: `retrieval/retrieval.py`

to recall **top_k** docs, based on the similarity from a certain **embedding model**, and use the mesh_dict json file just format in the last step.
 
```
#!/bin/bash

# Variables for the script
INPUT_FILE="/path/to/your/input_file.json"
OUTPUT_FILE="/path/to/your/output_file.json"
MESH_DESC_FILE="/path/to/mesh/desc_file.jsonl"
MODEL_PATH="/path/to/model"
QUERY_KEY1="query1_key"
QUERY_KEY2="query2_key"
TOP_K=4

python retrieval/retrieval.py \
	--input_file $INPUT_FILE \
	--output_file $OUTPUT_FILE \
	--mesh_desc_file $MESH_DESC_FILE \
	--model_path $MODEL_PATH \
	--query_key1 $QUERY_KEY1 \
	--query_key2 $QUERY_KEY2 \
	--Top_k $TOP_K

```


#### 4. Preference Selection: Determine the preference based on the MeSH terms from different Questions from different model inference.

Use the certain method to calculate the similarity between different Docs, and to choose the preference.

use the script: `mesh_score/cal_mesh_score.py`

```
python mesh_score/cal_mesh_score.py \
	--jsonl_file_path /path/to/input_file.jsonl \
	--output_file_path /path/to/output_file.jsonl \
	--mesh_desc_file /path/to/mesh_desc_file.jsonl \
	--database_path /path/to/database.db \
	--frequency_jsonl_path /path/to/frequency.jsonl

```


#### 5. Filtering: Filter out the preference data with particularly large deviations to reduce the risk of overfitting.

For the score values ​​of the two keys llama_score and bio_score, set a score_threshold as a filter, delete the data greater than the score_threshold, and output the remaining data in the same JSON format.

use the script: filtering/filter.py

```
python script.py \
	--input_file_path /path/to/input_file.jsonl \
	--output_file_path /path/to/output_file.jsonl \
	--score_threshold 0.5

```

----

### Do the DPO using trl framework.
----
```
pip install trl
```

use the scripts: trl/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py

```
CUDA_VISIBLE_DEVICES={} accelerate launch --num_processes 2 --main_process_port 29501 dpo_llama2.py \
    --model_name_or_path="/path/to/model" \
    --output_dir="saves/out_model"
```

----