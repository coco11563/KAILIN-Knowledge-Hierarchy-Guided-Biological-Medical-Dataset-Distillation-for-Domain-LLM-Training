## Prepare Data for Embedding model training and Do the SFT.

### Procedure of processing：

---------------

1. Gathering the raw data

In order to make up for the shortcomings of insufficient labeled datasets for comparative learning of Embedding models in the field of bioinformatics, we have extensively collected a large number of pair-wise data from PubMed, mainly in the form of title-abstract, title-full text, abstract-full text, etc.


2. Weak Consistency Filtering


We use a domain-specific Embedding model with good performance to evaluate the similarity of pair-wise data pairs and filter out low-similarity data.

Quick start: 

```
python Data-prepare/consistency_filtering.py --model_path /home/cxx/models/embedding/BioLORD-2023 \
    --folder_path /home/cxx/models/embedding/input_data/…… \
    --out_file_path /path/to/output/file.jsonl \
    --sim_threshold 0.75 \
    --key_x abstract \
    --key_y title
```

and the ouput file format:

```
outdata = {
                                'query': title,
                                'document': abstract,
                                'cos_sim': sim,
                                
                            }
```



3. Hard Negative Mining

We use FlagEmbedding framework to mine the hard negative data for better the Embedding model performance.

- Install FlagEmbedding from source
```
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip install  .
```
- Mine hard negatives 

	- Before Mining, you should install **Faiss** Package.
	```shell
	# CPU-only version
	$ conda install -c pytorch faiss-cpu=1.8.0

	# GPU(+CPU) version
	$ conda install -c pytorch -c nvidia faiss-gpu=1.8.0

	# GPU(+CPU) version with NVIDIA RAFT
	$ conda install -c pytorch -c nvidia -c rapidsai -c conda-forge faiss-gpu-raft=1.8.0
	```
		GPU version with  NVIDIA RAFT is recommended

	- Then use the following code to run the programme in the FlagEmbedding folder path.
	```
	python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
	--model_name_or_path BAAI/bge-base-en-v1.5 \
	--input_file toy_finetune_data.jsonl \
	--output_file toy_finetune_data_minedHN.jsonl \
	--range_for_sampling 2-200 \
	--negative_number 15 \
	--use_gpu_for_searching 

	#  where to sample negative. For example, 2-100 means sampling negative_number negatives from top2-top200 documents. You can set larger value to reduce the difficulty of negatives (e.g., set it 60-300 to sample negatives from top60-300 passages)

	```

---------------

### Procedure of SFT：

----
1. the train data format for SFT should be like:

```json
{"query": str, "pos": List[str], "neg":List[str]}
# the neg is from hard negatives mining.
```

2. run the training based on FlagEmbedding


```bash
torchrun --nproc_per_node {number of gpus} \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir {path to save model} \
--model_name_or_path BAAI/bge-large-zh-v1.5 \
--train_data ./toy_finetune_data.jsonl \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size {large batch size; set 1 for toy data} \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--save_steps 1000 \
--query_instruction_for_retrieval "" 
```


----


### Test your Embedding model perfomance.

----
We use mteb to test our Embedding model on BIOSSES, and other benchmarks.
```
pip install mteb
```

Do the test example:
```
python mteb_test/run_test.py \
	--model_path /path/to/model \
	--save_folder saves \
	--task BIOSSES
```

----
