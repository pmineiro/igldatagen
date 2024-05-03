#! /usr/bin/env python

from datasets import load_dataset
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def simulator_prompt(mood):
    return f"You are a user simulator.  A user has been presented a Question and an Answer. Simulate the user's next statement.  The user is {mood} with the Answer to the Question."

def encapsulate_qa(question, answer):
    return f"Question:\n{question}\n\nAnswer:\n{answer}\n"

def generate(*, messages, model, tokenizer, num_return_sequences):
    # https://huggingface.co/docs/transformers/v4.39.3/en/llm_tutorial#wrong-prompt
    model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    input_length = model_inputs.shape[1]
    generated_ids = model.generate(model_inputs, do_sample=True, max_new_tokens=100, num_return_sequences=num_return_sequences)
    return tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)

# https://chat.lmsys.org/?leaderboard
worse_model_id="Qwen/Qwen1.5-0.5B-Chat" # ELO not listed, so < 804
better_model_id="Qwen/Qwen1.5-32B-Chat" # ELO 1134

quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16,
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_quant_storage=torch.bfloat16,
                                        )

worse_model = AutoModelForCausalLM.from_pretrained(worse_model_id,
                                                   quantization_config=quantization_config,
                                                   torch_dtype=torch.bfloat16)
worse_tokenizer = AutoTokenizer.from_pretrained(worse_model_id, add_special_tokens=False)

better_model = AutoModelForCausalLM.from_pretrained(better_model_id,
                                                    quantization_config=quantization_config,
                                                    torch_dtype=torch.bfloat16)

better_tokenizer = AutoTokenizer.from_pretrained(better_model_id, add_special_tokens=False)

dataset = load_dataset('lmsys/chatbot_arena_conversations', token=os.environ.get('token', None))

with open('igldata.out', 'w') as file:
    for datum in tqdm(dataset['train']):
        question = datum['conversation_a'][0]['content']
        messages = [ { 'role': 'user', 'content': question } ]
        worse_answers = generate(messages=messages, model=worse_model, tokenizer=worse_tokenizer, num_return_sequences=4)
        better_answers = generate(messages=messages, model=better_model, tokenizer=better_tokenizer, num_return_sequences=1)

        # NB: we always use the better model to simulate user responses
    
        actions = []
        for answer in worse_answers:
            messages = [ { 'role': 'system', 'content': simulator_prompt("not satisfied") },
                         { 'role': 'user', 'content': encapsulate_qa(question, worse_answers[0]) },
                       ]
            worse_user_response = generate(messages=messages, model=better_model, tokenizer=better_tokenizer, num_return_sequences=1)
            actions.append({ 'action': answer,
                             'feedback': worse_user_response,
                             'reward': 0 })
    
        for answer in better_answers:
            messages = [ { 'role': 'system', 'content': simulator_prompt("satisfied") },
                         { 'role': 'user', 'content': encapsulate_qa(question, better_answers[0]) },
                       ]
            better_user_response = generate(messages=messages, model=better_model, tokenizer=better_tokenizer, num_return_sequences=1)
            actions.append({ 'action': answer,
                             'feedback': better_user_response,
                             'reward': 1 })
    
        igldatum = { 'question': question, 'actions': actions }
        print(json.dumps(igldatum), file=file, flush=True)
