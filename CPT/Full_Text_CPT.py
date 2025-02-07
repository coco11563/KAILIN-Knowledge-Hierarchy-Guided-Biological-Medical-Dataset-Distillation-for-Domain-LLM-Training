import os
import wandb
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

from itertools import islice
import json

            

max_seq_length = 8192  
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./model/llama3_70B",  # "unsloth/mistral-7b" for 16bit loading
    max_seq_length=max_seq_length,
    dtype=dtype,
    # load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    use_gradient_checkpointing = "unsloth",
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    random_state=3407,
    max_seq_length=max_seq_length,
)

dataset = Dataset.from_text("./CPT/CPT/10w_prompts_top4.txt")

EOS_TOKEN = tokenizer.eos_token
def formatting_func(example):
    return example["text"] + EOS_TOKEN

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    tokenizer = tokenizer,
    max_seq_length = max_seq_length,
    packing = True, # Packs short sequences together to save time!
    formatting_func = formatting_func,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,
        max_steps=600,

        warmup_ratio = 0.1,
        num_train_epochs = 1,
        save_strategy = 'steps',
        save_steps = 100,
        save_total_limit=10,

        learning_rate = 2e-5,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.00,
        lr_scheduler_type = "cosine",
        seed = 3407,
        
        output_dir = "10W_4C_Llama-3-70B_Base_Model",
    ),
)

# Initialize WandB
# wandb.init(project="10W_4C_Llama-3-70B_Base_Model", config={
#     # Your hyperparameters and other configs
#     "learning_rate": 2e-5,
#     "epochs": 1,
#     "lora_r": 32,
#     "lora_alpha": 16,
#     "lr_scheduler_type": "linear",
#     "per_device_train_batch_size": 2,
#     "gradient_accumulation_steps": 8,
#     "load_in_4bit":False,
#     
# })

trainer_stats = trainer.train()



if True: model.save_pretrained_merged("10W_4C_Llama-3-70B_Base_Model", tokenizer, save_method = "merged_16bit",)