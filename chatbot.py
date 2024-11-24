import torch
import numpy as np
import time
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='This is a demonstration program')

parser.add_argument('-batch_size', type=str, required=True, help='Please provide a batch_size')
args = parser.parse_args()

print(f'batch size: {args.batch_size}')

block_size = 128
batch_size = int(args.batch_size)
learning_rate = 2e-5
eval_iters = 100
n_embed = 384
n_head = 8
n_layers = 8
dropout = 0.2

chars = ""
with open("dataset/vocab.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)

string_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_string = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

class Head(nn.Module):
    """
    one head of self-attention
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # 将tril注册为一个缓冲区，他们是模型的一部分，用作mask，但不会在训练中参与梯度更新
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))# 其实就是建立一个下三角矩阵

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # input of size: (batch, time-step, channels)
        # output of size: (batch, time-step, head_size)
        _,T,_ = x.shape
        k = self.key(x) # shape:(B,T,hs)
        q = self.query(x) # shape:(B,T,hs)
        # compute attention scores
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,hs) * (B,hs,T) -> (B,T,T)
        # masked_fill:根据mask来填充张量中的某些位置，若是对应位置为True,则将该位置填充为指定的值
        # torch.tril():将某个张量转化为下三角矩阵，上三角部分置为0
        # 用-inf替换掉wei中与tril的上三角部分对应的元素
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float('-inf')
        )
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
    

class MultiHeadAttention(nn.Module):
    """
    multiple heads of self-attention in parallel
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(out)
        return out    
    
class FeedForward(nn.Module):
    """
    a simple linear layer followed by a non-linearity
    """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer Block: communication followed by computation
    """
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffd(x)
        x = self.ln2(x + y)
        return x
    
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        _,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)


        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:,-block_size:] # negative: starts counting from the end of the array
            logits, _ = self.forward(index_cond)
            logits = logits[:, -1, :] # take the last time step
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index_cond, index_next), dim=1)
        return index

model = GPTLanguageModel(vocab_size)
print("loading model parameters...")
with open('./model/model-01.pkl','rb') as f:
    model = pickle.load(f)
print("loaded successfully!")
m = model.to(device)


while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(
        encode(prompt),
        dtype=torch.long,
        device=device
    )
    # generated_charts = decode(
    #     m.generate(
    #         context.unsqueeze(0),
    #         max_new_tokens=150
    #     )[0].tolist()
    # )
    a = m.generate(
        context.unsqueeze(0),
        max_new_tokens = 150
    )
    generated_charts = decode(a[0].tolist())
    print(f"m.generate:\n{a}")
    print(f"len(m.generate):\n{len(a[0].tolist())}")
    print(f"Completion:\n{generated_charts}")


