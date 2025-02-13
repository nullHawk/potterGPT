import torch
from dataclasses import dataclass

@dataclass
class Config:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.block_size = 256 # context-length
        self.batch_size = 64 # mini-batch size
        self.vocab_size = self.tokenizer.VOCAB_SIZE
        self.n_embed = 256
        self.n_heads = 8
        self.head_size = self.n_embed // self.n_heads # computes to 384/6=64 or 128/4=32 or 256/8
        
        self.n_layers = 3
        
        self.train_iters = 10_000
        self.val_iters = 1000
        self.lr = 3e-4
        
        self.attn_dropout = 0.1
        self.block_dropout = 0.1
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
