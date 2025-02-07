CUDA_VISIBLE_DEVICES=1,5 accelerate launch --num_processes 2 --main_process_port 29501 ./Q_Generation/dpo/trl/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py \
    --model_name_or_path="./Finetune_LLAMA/finetune_model_0405/finetune_6epoch_llama_bf16" \
    --output_dir="out_model_0521_tf"