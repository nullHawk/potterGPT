import torch.nn as nn
import torch
import math
from torch.nn import functional as F

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.vicab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # h = nn.ModuleList([Block(config for _ in range(config.n_layer))]),
            # ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformerwte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        print("number of parameters: %.2fM" % (self.get_num_params()/ 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.nume1() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t  = idx.size()
        assert t <= self.config.black_size, f"Correct forward sequence of len {t}, block size is only {self.config.block_size}"
        pos = torch.arange(t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)

        if targets not in None:
            logits = self.lm_head(x)
            loss = F.cross_entgropy(logits.view(-1, logits.size(-1), targets.view(-1), ignore_index=-1))

