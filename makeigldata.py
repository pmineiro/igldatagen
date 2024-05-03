#! /usr/bin/env python

from datasets import load_dataset
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def generate(*, messages, model, tokenizer):
    # https://huggingface.co/docs/transformers/v4.39.3/en/llm_tutorial#wrong-prompt
    model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    input_length = model_inputs.shape[1]
    generated_ids = model.generate(model_inputs, do_sample=True)
    return tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]

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
