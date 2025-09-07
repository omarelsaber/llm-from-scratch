# src/embeddings.py
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, context_length=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        b, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        token_embeds = self.token_embedding(input_ids)          # [b, seq_len, dim]
        pos_embeds = self.pos_embedding(positions)              # [1, seq_len, dim]
        return token_embeds + pos_embeds
