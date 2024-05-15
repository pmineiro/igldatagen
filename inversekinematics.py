#! /usr/bin/env python

from Agents import Adapter, YesNoAgent
import json
from peft import LoraConfig, TaskType
from ProgressPrinter import ProgressPrinter
from random import Random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from util import CTSig, make_optimizer

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

def generate_data(handle):
    Object = lambda **kwargs: type("Object", (), kwargs)()
    rgen = Random(8675309)
    for line in handle:
        obj = json.loads(line)
        question = obj['question']
        rgen.shuffle(obj['actions'])
        for n, action in enumerate(obj['actions']):
            yield Object(reward = action['reward'],
                         prompts = [ format_for_classification(question=question, answer=other['action'], feedback=action['feedback'][0], ctsig=ctsig) for other in obj['actions'] ],
                         labels = [ m==n for m, _ in enumerate(obj['actions']) ],
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
adapters = [ Adapter(cls=YesNoAgent, suffix='ik', model_id=None, prototype=prototype, empty_cache=False, init_kwargs={}), ]
for adapter in adapters:
    adapter.load_pretrained(model=model)
model.enable_adapters()
model.set_adapter([ n for adapter in adapters for n in adapter.get_all_names() ])

clip = 1.0
clip_lam = lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), clip, norm_type=torch.inf)
alpha = 1000
optimizer, pcnt = make_optimizer(model, alpha)

wrappers = { adapter.suffix: adapter.init_wrapper(model=model, tokenizer=tokenizer, optimizer=optimizer, clip=clip_lam) for adapter in adapters }
metrics = [ 'posloss', 'posacc', 'negloss', 'negacc' ]

save_every = 1000
with ProgressPrinter(*metrics) as progress, open('igldata.out', 'r') as handle:
    for n, datum in enumerate(low_mem_shuffle(generate_data(handle))):
        scores = wrappers['ik'].score(datum.prompts, micro_batch_size=4)
        best_score_index = max(enumerate(scores), key=lambda v:v[1])[0]
        updates = [ (msg, "Yes" if label else "No", 1) for msg, label in zip(datum.prompts, datum.labels) ]
        loss = wrappers['ik'].learn(updates, sync_each_batch=False, empty_cache=False, micro_batch_size=1)

        if datum.reward:
            pos_acc, neg_acc = datum.labels[best_score_index], None
            pos_loss, neg_loss = loss, None
        else:
            neg_acc, pos_acc = datum.labels[best_score_index], None
            neg_loss, pos_loss = loss, None

        progress.addobs(pos_loss, pos_acc, neg_loss, neg_acc)

        if n >= save_every and n % save_every == 0:
            progress.autoprint = False
            progress.print()
            wrappers['ik'].save_pretrained(pathspec = [ "save", str(n) ])
