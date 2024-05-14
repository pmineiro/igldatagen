class CTSig(object):
    def __init__(self, *, tokenizer):
        import re

        super().__init__()
        try:
            messages = [ { 'role': 'system', 'content': 'wazzup' }, { 'role': 'user', 'content': 'hey' } ]
            tokenizer.apply_chat_template(messages, tokenize=False)
            supports_system = True
        except:
            supports_system = False

        self._supports_system = supports_system

    @property
    def supports_system(self):
        return self._supports_system

    def render_with_system(self, *, system, user):
        if self.supports_system:
            return [ { 'role': 'system', 'content': system }, { 'role': 'user', 'content': user } ]
        else:
            return [ { 'role': 'user', 'content': f'{system}\n\n{user}' } ]

    def __str__(self):
        return str((self.supports_system,))

def make_optimizer(model, alpha):
    # TODO (?): https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html#disable-foreach-in-the-optimizer

    params = [ p for p in model.parameters() if p.requires_grad ]

    if alpha < 0:
        import bitsandbytes as bnb
        optim = bnb.optim.Adam8bit(params, lr=-alpha)
    elif alpha < 1:
        import torch
        optim = torch.optim.Adam(params, lr=alpha)
    else:
        import parameterfree
        optim = parameterfree.COCOB(params, alpha=alpha)

    return optim, sum(p.numel() for p in params)

