from itertools import islice
import json

jsonl_path = "./Mesh_Sim/data/preference_data.json"
output_file = "./10w_prompts_top4.txt"

TOP = 4

def read_chunks(file_path, chunk_size=100):
    with open(file_path, 'r') as file:
        while True:
            chunk = list(islice(file, chunk_size))
            if not chunk:
                break
            yield chunk

def create_prompt(data):
    prompts = []
    for item in data:
        item_data = json.loads(item)
        raw_title = item_data.get("title")
        raw_title = raw_title.replace('\n', ' ')
        # print(raw_title)
        raw_context = item_data.get("raw_context")
        raw_context = raw_context.replace('\n', ' ')
        context = item_data.get("llama_recall_context")[:TOP]
        # context = context.replace('\n', ' ')
        # print((context[0]))
        question = item_data.get("llama_q")
        # question = question.replace('\n', ' ')
        
        prompt = f"After reading the paper ```{raw_title}:{raw_context}```, I searched for related materials ```{context}``` and pondered the following scientific question ```{question}```."

        prompt = prompt.replace('\n', ' ')
        prompts.append(prompt)
    if len(prompts) > 100:
        print(len(prompts))
    return prompts

with open(output_file, 'w') as f:
    num = 0
    for chunk in read_chunks(jsonl_path, 100):
        prompts = create_prompt(chunk)
        for prompt in prompts:
            num += 1
            f.write(prompt + "\n")

            