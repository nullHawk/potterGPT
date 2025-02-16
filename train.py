from model import CharacterLevelTokenizer, Config, PotterGPT
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

torch.manual_seed(1357)
with open('data/harry_potter_data', 'r', encoding='utf-8') as f:
    data = f.read()

class ShakespeareDataset:
    def __init__(self,Config, is_test=False) -> None:
        self.tokenizer = CharacterLevelTokenizer(data)
        self.is_test = is_test
        self.full_data = self.tokenizer.encode(self.tokenizer.data)
        if self.is_test:
            self.data = self.full_data[int(0.9*len(self.full_data)):]
        else:
            self.data = self.full_data[:int(0.9*len(self.full_data))]
        self.block_size = Config.block_size
        self.batch_size = Config.batch_size

    def __len__(self) -> int:
        return len(self.data)

    def get_block_size(self) -> int:
        return self.block_size

    def get_vocab_size(self) -> int:
        return self.tokenizer.VOCAB_SIZE

    def get(self):
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        return x,y

tokenizer = CharacterLevelTokenizer(data)

#Training

train_ds = ShakespeareDataset(Config)
val_ds = ShakespeareDataset(Config, is_test=True)

lm = PotterGPT(Config)
lm = lm.to(device=Config.device)

optim = torch.optim.Adam(lm.parameters(), lr=Config.lr)

def loss_fn(logits, targets):
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits, targets)
    return loss

def train_N_iters():
    lm.train()
    train_step_losses = []
    for batch in tqdm(range(Config.train_iters)):
        optim.zero_grad()
        inputs, targets = train_ds.get()
        inputs, targets = inputs.to(device=Config.device), targets.to(device=Config.device)
        logits = lm(inputs)
        loss = loss_fn(logits,targets)
        loss.backward()
        optim.step()
        train_step_losses.append(loss.item())
        
        if batch%(Config.train_iters//10)==0 or batch==Config.train_iters-1:
            print(f"batch {batch} train step loss: {loss.item()}")
        
        del inputs, targets, loss, logits
        
    return train_step_losses
    
@torch.no_grad()
def valid_N_iters():
    lm.eval()
    val_step_losses = []
    for batch in tqdm(range(Config.val_iters)):
        inputs, targets = val_ds.get()
        inputs, targets = inputs.to(device=Config.device), targets.to(device=Config.device)
        logits = lm(inputs)
        loss = loss_fn(logits,targets)
        val_step_losses.append(loss.item())
        
        if batch%(Config.val_iters//10)==0 or batch==Config.val_iters-1:
            print(f"batch {batch} valid step loss: {loss.item()}")
        
        del inputs, targets, loss, logits
    
    return val_step_losses

def save_lm():
    state_dict = lm.state_dict()
    save_path = Path('./').resolve() / 'potterGPT'
    save_path.mkdir(exist_ok=True)
    model_path = save_path / f'potterGPT.pth'
    torch.save(state_dict, model_path)

def train_lm():
    train_losses = train_N_iters()
    valid_losses = valid_N_iters()
    save_lm()
    return train_losses, valid_losses

tl, vl = train_lm()

plt.plot(tl,label='train loss',color='orange')
plt.plot(vl,label='valid loss',color='blue')
plt.title('Potter GPT Losses')
plt.legend()
plt.show()

generated_texts = []
for length in [100,300,500,700,1000]:
    generated = lm.generate(
    torch.zeros((1,1),dtype=torch.long,device=Config.device), # initial context 0
    total=length
)
    generated = tokenizer.decode(generated[0])
    text=f'generated ({length} tokens)\n{"="*50}\n{generated}\n{"="*50}\n\n'
    generated_texts.append(text)
    print(text)