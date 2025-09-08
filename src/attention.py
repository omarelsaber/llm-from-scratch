import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(size: int) -> torch.Tensor:
    """
    Create a causal mask for self-attention (upper-triangular).
    Ensures that position i can only attend to positions <= i.
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)  # upper triangle
    return mask == 0  # convert to boolean mask


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Final projection
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()  # Batch, Time, Channels

        # Project to Q, K, V
        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, nh, T, T)

        # Apply causal mask
        mask = causal_mask(T).to(x.device)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        out = attn_weights @ V  # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # merge heads
        out = self.W_o(out)  # final linear

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Attention + Residual
        x = x + self.attn(self.ln1(x))
        # Feed-forward + Residual
        x = x + self.ff(self.ln2(x))
        return x
