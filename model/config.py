import torch
from .tokenizer import CharacterLevelTokenizer
from dataclasses import dataclass

with open('data/harry_potter_data', 'r', encoding='utf-8') as f:
    data = f.read()

@dataclass
class Config:
    tokenizer = CharacterLevelTokenizer(data)
    block_size = 256 # context-length
    batch_size = 64 # mini-batch size
    vocab_size = tokenizer.VOCAB_SIZE
    n_embed = 256
    n_heads = 8
    head_size =n_embed //n_heads # computes to 384/6=64 or 128/4=32 or 256/8
        
    n_layers = 3
        
    train_iters = 10_000
    val_iters = 1000
    lr = 3e-4
        
    attn_dropout = 0.1
    block_dropout = 0.1
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
