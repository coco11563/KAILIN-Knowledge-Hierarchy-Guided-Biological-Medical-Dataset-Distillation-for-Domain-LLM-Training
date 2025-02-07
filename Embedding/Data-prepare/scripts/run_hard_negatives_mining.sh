python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
	--model_name_or_path BAAI/bge-base-en-v1.5 \
	--input_file toy_finetune_data.jsonl \
	--output_file toy_finetune_data_minedHN.jsonl \
	--range_for_sampling 2-200 \
	--negative_number 15 \
	--use_gpu_for_searching 

#  where to sample negative. For example, 2-100 means sampling negative_number negatives from top2-top200 documents. You can set larger value to reduce the difficulty of negatives (e.g., set it 60-300 to sample negatives from top60-300 passages)