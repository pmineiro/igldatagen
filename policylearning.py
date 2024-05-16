#! /usr/bin/env python

from Agents import Adapter, YesNoAgent
from math import exp, sqrt
from peft import LoraConfig, TaskType
from ProgressPrinter import ProgressPrinter
from random import Random
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from util import CTSig, make_optimizer
import gzip
import json
import sys
import torch

if len(sys.argv) < 6:
    print(f'{sys.argv[0]} ik_save_dir skip_initial_count beta sigma gamma0', file=sys.stderr)
    exit(1)

ik_save_dir = sys.argv[1]
skip_initial_count = int(sys.argv[2])
beta = float(sys.argv[3])
sigma = float(sys.argv[4])
gamma0 = float(sys.argv[5])

print(f'ik_save_dir={ik_save_dir} skip_initial_count = {skip_initial_count} beta={beta} sigma={sigma} gamma0={gamma0}', flush=True)

#================================================

def format_for_fhat(question, answer, ctsig):
    import re 

    question = ' '.join(question.split())
    answer = ' '.join(answer.split())
    system = f"You are an Answer evaluating agent.  Given a User's Question and an Answer: assess if the Answer is good.  Respond with Yes or No only."
    user = f"User's Question:\n{question}\n\nAnswer:\n{answer}\n\nRespond with Yes or No only."

    return ctsig.render_with_system(system=system, user=user)

def format_for_classification(question, answer, feedback, ctsig):
    import re 

    question = ' '.join(question.split())
    answer = ' '.join(answer.split())
    feedback = re.sub(r'^User:', '', feedback)
    feedback = feedback.split('\n\n', maxsplit=2)[0]
    feedback = ' '.join(feedback.split())
    system = f"You are a conversation evaluating agent.  Given a User's Question, an Answer, and the User's Feedback: determine if the User's Feedback is consistent with Answer.  Respond with Yes or No only."
    user = f"User's Question:\n{question}\n\nAnswer:\n{answer}\n\nUser's Feedback:\n{feedback}\n\nRespond with Yes or No only."

    return ctsig.render_with_system(system=system, user=user)

def generate_data(handle, *, skip):
    Object = lambda **kwargs: type("Object", (), kwargs)()
    for lineno, line in enumerate(handle):
        if lineno < skip:
            continue

        obj = json.loads(line)
        question = obj['question']
        yield Object(rewards = [ a['reward'] for a in obj['actions'] ],
                     ik_prompts = [ format_for_classification(question, answer=a['action'], feedback=a['feedback'][0], ctsig=ctsig) for a in obj['actions'] ],
                     fhat_prompts = [ format_for_fhat(question, answer=a['action'], ctsig=ctsig) for a in obj['actions'] ],
                    )

def low_mem_shuffle(gen):
    rgen = Random(2112)
    buf = [None]*10000

    for v in gen:
        index = rgen.randrange(len(buf))
        if buf[index] is not None:
            yield buf[index][0]
        buf[index] = (v,)

    yield from [ v[0] for v in buf if v is not None ]

# https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora#training
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16,
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_quant_storage=torch.bfloat16,
                                        )

base_model_id="meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base_model_id,
                                             quantization_config=quantization_config,
                                             torch_dtype=torch.bfloat16)

# begin peft.prepare_model_for_kbit_training ... it doesn't handle torch.bfloat16 properly, so put the important stuff inline.
for name, param in model.named_parameters():
    # freeze base model's layers
    param.requires_grad = False
model.gradient_checkpointing_enable() # use_reentrant=False (?)
# When having `use_reentrant=False` + gradient_checkpointing, there is no need for this hack
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)
model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
# end peft.prepare_model_for_kbit_training

tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_special_tokens=False)
assert tokenizer.eos_token is not None
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
ctsig = CTSig(tokenizer=tokenizer)

proto_target_modules = ['qkv_proj'] if 'Phi-3' in base_model_id else ['q_proj', 'v_proj']
prototype = LoraConfig(r=1, task_type=TaskType.CAUSAL_LM, target_modules=proto_target_modules, use_rslora=True)
adapters = [ Adapter(cls=YesNoAgent, suffix='ik', model_id=ik_save_dir, prototype=prototype, empty_cache=False, init_kwargs={}),
             Adapter(cls=YesNoAgent, suffix='fhat', model_id=None, prototype=prototype, empty_cache=False, init_kwargs={}),
           ]
for adapter in adapters:
    adapter.load_pretrained(model=model)
model.enable_adapters()
model.set_adapter([ n for adapter in adapters for n in adapter.get_all_names() ])

clip = 1.0
clip_lam = lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), clip, norm_type=torch.inf)
alpha = 1000
optimizer, pcnt = make_optimizer(model, alpha)

wrappers = { adapter.suffix: adapter.init_wrapper(model=model, tokenizer=tokenizer, optimizer=optimizer, clip=clip_lam) for adapter in adapters }
metrics = [ 'fhatloss', 'inferredreward', 'truereward' ]

save_every = 1000
with ProgressPrinter(*metrics) as progress, gzip.open('igldata.out.gz', 'r') as handle:
    rgen = Random(8675309)
    for n, datum in enumerate(low_mem_shuffle(generate_data(handle, skip=skip_initial_count))):
        with torch.no_grad():
            fhats = wrappers['fhat'].score(datum.fhat_prompts, micro_batch_size=8)
            maxfhat = max(fhats)
            gamma = sqrt(1 + n) / gamma0
            rawigw = [ 1 / (len(datum.fhat_prompts) + gamma * (maxfhat - fhat)) for fhat in fhats ]
            sumrawigw = sum(rawigw)
            igw = [ rawp / sumrawigw for rawp in rawigw ]
            action = rgen.choices(list(range(len(datum.fhat_prompts))), weights=igw)[0]

            ik_score = wrappers['ik'].score([ datum.ik_prompts[action] ], micro_batch_size=1)[0]
            inferred_reward = max(0, min(1, (exp(ik_score) - beta) / sigma))
            true_reward = datum.rewards[action]

        updates = [ (msg, label, weight) 
                    for msg in (datum.fhat_prompts[action],)
                    for (label, weight) in (("Yes", inferred_reward), ("No", 1 - inferred_reward),)
                  ]
        loss = wrappers['fhat'].learn(updates, sync_each_batch=False, empty_cache=False, micro_batch_size=1)

        progress.addobs(loss, inferred_reward, true_reward)

        if n >= save_every and n % save_every == 0:
            progress.autoprint = False
            progress.print()
            wrappers['fhat'].save_pretrained(pathspec = [ "save", str(n) ])
