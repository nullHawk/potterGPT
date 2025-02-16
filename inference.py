import torch
from model import PotterGPT, Config 
from tokenizers import Tokenizer
from dataclasses import dataclass

model_path = 'potterGPT/potterGPT.pth'

tokenizer = Tokenizer.from_file('tokenizer/potter.json')



lm = PotterGPT(Config)
state_dict = torch.load(model_path, map_location='cpu')
lm.load_state_dict(state_dict)

generated_texts = []
for length in [1000]:
    generated = lm.generate(
    torch.zeros((1,1),dtype=torch.long,device='cpu') + 61, # initial context 0, 61 is \n
    total=length
)
    generated = tokenizer.decode(generated[0].cpu().numpy())
    text=f'generated ({length} tokens)\n{"="*50}\n{generated}\n{"="*50}\n\n'
    generated_texts.append(text)

with open('generated.txt','w') as f:
    for text in generated_texts:
        f.write(text)