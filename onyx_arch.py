import torch
import torch.nn as nn
from torch.nn import functional as F

# Configuration matches the training script
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.0 
block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x):
        # Using MultiheadAttention for deep pattern recognition
        attn_out, _ = self.sa(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x

class OnyxModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        # Truncate to block_size to prevent index errors
        idx = idx[:, -block_size:] 
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits