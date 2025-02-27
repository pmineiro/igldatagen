import torch

class Adapter(object):
    def __init__(self, *, cls, suffix, model_id, prototype, empty_cache, init_kwargs):
        self._cls = cls
        self._suffix = suffix
        self._model_id = model_id
        self._prototype = prototype
        self._empty_cache = empty_cache
        self._init_kwargs = init_kwargs

    @property
    def model_id(self):
        return self._model_id

    @property
    def prototype(self):
        return self._prototype

    @property
    def suffix(self):
        return self._suffix

    def load_pretrained(self, *, model):
        self._cls.load_adapters(model=model, adapter=self)

    def get_all_names(self):
        return self._cls.get_all_adapter_names(adapter=self)

    def init_wrapper(self, *, model, tokenizer, optimizer, clip):
        obj =  self._cls(model=model, tokenizer=tokenizer, adapter_suffix=self._suffix, optimizer=optimizer, clip=clip, **self._init_kwargs)
        return EmptyCacheWrapper(obj=obj) if self._empty_cache else obj

class BaseAgent(torch.nn.Module):
    @classmethod
    def load_adapters(cls, *, model, adapter, prefixes):
        for prefix in prefixes:
            adapter_name = f'{prefix}_{adapter.suffix}'
            if adapter.model_id and not prefix.startswith('_'):
                model.load_adapter(adapter.model_id, adapter_name)
            else:
                from copy import deepcopy
                ref_config = deepcopy(adapter.prototype)
                model.add_adapter(ref_config, adapter_name)

    @classmethod
    def get_all_adapter_names(cls, *, adapter):
        pass

    @classmethod
    def empty_cache(cls):
        torch.cuda.empty_cache()

    def __init__(self, *, model, tokenizer, adapter_suffix, generate_kwargs):
        super().__init__()
        self._transformer = model
        self._tokenizer = tokenizer
        self._adapter_suffix = adapter_suffix
        self._generate_kwargs = generate_kwargs

    def set_adapter(self, *, prefix):
        self._transformer.set_adapter(f'{prefix}_{self._adapter_suffix}')

    def _batch_logp(self, logits, labels):
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        normalized = logits.log_softmax(-1)
        res = loss_fct(normalized.view(-1, normalized.size(-1)), labels.view(-1)).view_as(labels)
        return -res.view(labels.size(0), -1).sum(dim=1)

    def _casual_lm_seq2seq(self, prefix_lens, full_toks):
        combo_input_ids = full_toks.input_ids
        combo_attention_mask = full_toks.attention_mask
        combo_labels = combo_input_ids.clone()
        combo_labels[combo_attention_mask == 0] = -100
        for n, pl in enumerate(prefix_lens):
            combo_labels[n, :pl] = -100
        combo_labels[:, :-1] = combo_labels.clone()[:, 1:]
        combo_labels[:, -1] = -100
        return type('',(object,),{"input_ids": combo_input_ids.to(self._transformer.device),
                                  "attention_mask": combo_attention_mask.to(self._transformer.device),
                                  "labels": combo_labels.to(self._transformer.device)
                                 })()

    def _generate_one(self, raw_msg, *, num_return_sequences=1, num_beams=None):
        msg = raw_msg if raw_msg is not None else [ { 'role': 'user', 'content': 'hi' } ]

        do_beam = self._generate_kwargs.get('do_beam', False)
        max_new_tokens = self._generate_kwargs.get('max_new_tokens', 100)
        diverse_beam = self._generate_kwargs.get('diverse_beam', False)
        explore = self._generate_kwargs.get('explore', True)

        with torch.no_grad():
            self.eval()
            if num_beams is None:
                num_beams = 5 * num_return_sequences
            if do_beam:
                kwargs = { 'num_beams': num_beams, 'do_sample': explore and not diverse_beam, 'early_stopping': not diverse_beam }
                if diverse_beam and num_beams > 1:
                    kwargs.update({ 'num_beam_groups': num_beams, 'diversity_penalty': 1.0 })
            else:
                kwargs = { 'do_sample': explore }

            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            kwargs.update({ 'temperature': 0.7 if kwargs['do_sample'] else None,
                            'top_k': None,
                            'top_p': 0.9 if kwargs['do_sample'] else None,
                            'synced_gpus': isinstance(self._transformer, FSDP),
                          })

            # https://huggingface.co/docs/transformers/v4.39.3/en/llm_tutorial#wrong-prompt

            model_inputs = self._tokenizer.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt").to(self._transformer.device)
            input_length = model_inputs.shape[1]

            # wtf ... (?)
            # inspired by https://github.com/pytorch/pytorch/issues/82461
            # but it's totally unclear why this is necessary
            if isinstance(self._transformer, FSDP):
                self._transformer(model_inputs, attention_mask=None)

            generated_ids = self._transformer.generate(model_inputs,
                                                       attention_mask=None,
                                                       max_new_tokens=max_new_tokens,
                                                       num_return_sequences=num_return_sequences,
                                                       pad_token_id=self._tokenizer.eos_token_id,
                                                       **kwargs)
            return self._tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)

    def learn(self, update_data, *, sync_each_batch, empty_cache, micro_batch_size):
        from more_itertools import batched

        # NB: does not work if update_data is empty
        assert update_data

        optimizer = self._optimizer
        clip = self._clip

        self.eval() # NB: dropout etc. is empirically no good with PEFT

        # NB: sadly, gradient accumulation with FSDP can increase memory usage
        # https://discuss.pytorch.org/t/vram-usage-increase-with-more-gradient-accumulation-steps/180729
        # https://huggingface.co/docs/accelerate/v0.29.3/en/concept_guides/gradient_synchronization#nosync-requires-additional-gpu-memory-when-using-fsdp

        avloss = 0
        scalefac = sum(int(datum is not None) for datum in update_data) # NB: use None as padding to ensure identical control flow

        if empty_cache:
            torch.cuda.empty_cache()
        optimizer.zero_grad(set_to_none=True)

        micro_batches = list(batched(update_data, micro_batch_size))
        for n, micro_batch in enumerate(micro_batches):
            if n + 1 < len(micro_batches) and not sync_each_batch:
                import contextlib
                manager = self._transformer.no_sync if callable(getattr(self._transformer, 'no_sync', None)) else contextlib.nullcontext
                with manager():
                    loss = self(micro_batch, scalefac=scalefac)
                    loss.backward()
                    avloss += loss.item()
                    del loss
            else:
                loss = self(micro_batch, scalefac=scalefac)
                loss.backward()
                avloss += loss.item()
                del loss

                if clip is not None:
                    clip() # for FSDP: clip = lambda: model.clip_grad_norm_(value)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if empty_cache:
                    torch.cuda.empty_cache()

        return avloss

    def set_ref_to_trained(self, *, weight):
        pass

    def save_pretrained(self, *, pathspec):
        pass

class YesNoAgent(BaseAgent):
    @classmethod
    def load_adapters(cls, *, model, adapter):
        super().load_adapters(model=model, adapter=adapter, prefixes=("trained",))

    @classmethod
    def get_all_adapter_names(cls, *, adapter):
        return [ f'{prefix}_{adapter.suffix}' for prefix in ("trained",) ]

    def __init__(self, *, model, tokenizer, adapter_suffix, optimizer, clip):
        super().__init__(model=model, tokenizer=tokenizer, adapter_suffix=adapter_suffix, generate_kwargs={})
        self._optimizer = optimizer
        self._clip = clip
        self._yes = self._tokenizer(["Yes"], add_special_tokens=False).input_ids[0][0]
        assert 'Yes' == self._tokenizer.batch_decode([ self._yes ])[0]
        self._no = self._tokenizer(["No"], add_special_tokens=False).input_ids[0][0]
        assert 'No' == self._tokenizer.batch_decode([ self._no ])[0]

    def _yes_no_logprobs(self, msgs):
        with torch.no_grad():
            self._tokenizer.padding_side = 'left'
            toks = self._tokenizer([ self._tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) for msg in msgs ],
                                   return_tensors='pt', padding=True).to(self._transformer.device)

        logits = self._transformer(input_ids=toks.input_ids,
                                   attention_mask=None if len(msgs) == 1 else toks.attention_mask
                                  ).logits[:,-1,[self._yes,self._no]]
        return torch.nn.functional.log_softmax(logits, dim=1)

    def score(self, raw_msgs, *, micro_batch_size):
        from more_itertools import batched

        # FSDP requires identical control flow, so use None as a padding input
        msgs = [ m if m is not None else [ { 'role': 'user', 'content': 'hi' } ] for m in raw_msgs ]

        self.set_adapter(prefix="trained")
        with torch.no_grad():
            all_scores = [ score
                           for micro_batch in batched(msgs, micro_batch_size)
                           for score in self._yes_no_logprobs(micro_batch)[:,0].tolist()
                         ]
            return [ score for raw_msg, score in zip(raw_msgs, all_scores) if raw_msg is not None ]

    def forward(self, raw_yesno_data, *, scalefac):
        # FSDP requires identical control flow, so generate a zero reward input that will call forward() but produce a loss of 0 with no gradient
        yesno_data = [ ([ { 'role': 'user', 'content': 'a' } ], 'Yes', 0) if p is None else p for p in raw_yesno_data ]

        self.set_adapter(prefix="trained")
        logprobs = self._yes_no_logprobs([ prefix for prefix, _, _ in yesno_data ])

        with torch.no_grad():
            labels = torch.Tensor([ 0 if label == 'Yes' else 1 for _, label, _ in yesno_data ]).long().to(logprobs.device)

        neglogplabels = torch.nn.functional.nll_loss(logprobs, labels, reduction='none')

        # pg loss
        rewards = torch.Tensor([ reward for _, _, reward in yesno_data ]).to(self._transformer.device)
        return torch.dot(rewards, neglogplabels) / max(1, scalefac)

    def save_pretrained(self, *, pathspec):
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

            super().save_pretrained(pathspec=pathspec)
            try:
                model_id = '_'.join([pathspec[0], self._adapter_suffix, pathspec[1]])
                self.set_adapter(prefix="trained")
                # TODO: get rid of warning
                self._transformer.save_pretrained(model_id)
            except Exception as e:
                print(f'an (ignored) exception has occured while attempting model save: {e}', flush=True)
