import torch
import os
from model import PotterGPT, Config, CharacterLevelTokenizer
from tokenizers import Tokenizer
from dataclasses import dataclass

model_path = 'potterGPT/potterGPT.pth'
with open('data/harry_potter_data', 'r', encoding='utf-8') as f:
    data = f.read()


tokenizer = CharacterLevelTokenizer(data)



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

print(generated_texts[0])

os.makedirs('output', exist_ok=True)
with open('output/generated.txt', 'w+') as f:
    for text in generated_texts:
        f.write(text)