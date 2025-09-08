# src/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(seq_len: int, device=None, dtype=torch.bool) -> torch.Tensor:
    """
    Upper-triangular mask (True above diagonal) to block attention to future tokens.
    Shape: (T, T)
    """
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=1)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention with optional FlashAttention path (torch>=2.0).
    Expects Q,K,V of shape (B, H, T, Dh)
    """
    def __init__(self, dropout: float = 0.0, use_flash: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if self.use_flash:
            # Flash path (handles masking internally). attn_mask: True means "blocked".
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal
            )

        # Manual path
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)  # (B,H,T,T)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        elif is_causal:
            T = q.size(-2)
            cm = causal_mask(T, device=q.device)  # (T,T)
            scores = scores.masked_fill(cm, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return torch.matmul(weights, v)  # (B,H,T,Dh)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (causal).
    x: (B, T, C) where C = d_model
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn = ScaledDotProductAttention(dropout=dropout, use_flash=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # (B,T,3C)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):
            # (B,T,C) -> (B,H,T,Dh)
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # causal mask (T,T) -> broadcast to (B,H,T,T)
        if attn_mask is None:
            cm = causal_mask(T, device=x.device)  # True where we block
            y = self.attn(q, k, v, attn_mask=cm, is_causal=False)
        else:
            y = self.attn(q, k, v, attn_mask=attn_mask, is_causal=False)

        # (B,H,T,Dh) -> (B,T,C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y


class FeedForward(nn.Module):
    """Simple MLP block used inside Transformer blocks."""
    def __init__(self, d_model: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Minimal Transformer decoder block (Pre-LN):
    x -> LN -> MHA -> + -> LN -> MLP -> +
    """
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
