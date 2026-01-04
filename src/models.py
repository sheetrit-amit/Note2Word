from __future__ import annotations

import torch
import torch.nn as nn


class TextMelodyRNNStatic(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, melody_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(
            input_size=embed_dim + melody_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor, melody_static: torch.Tensor):
        x = self.embedding(tokens)
        melody = melody_static.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, melody], dim=-1)
        out, _ = self.rnn(x)
        logits = self.fc(out)
        return logits


class TextMelodyRNNTemporal(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, melody_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(
            input_size=embed_dim + melody_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor, melody_seq: torch.Tensor):
        x = self.embedding(tokens)
        melody = melody_seq[:, : x.size(1), :]
        if melody.size(1) < x.size(1):
            pad = torch.zeros(x.size(0), x.size(1) - melody.size(1), melody.size(2), device=melody.device)
            melody = torch.cat([melody, pad], dim=1)
        x = torch.cat([x, melody], dim=-1)
        out, _ = self.rnn(x)
        logits = self.fc(out)
        return logits


def init_with_embeddings(model: nn.Module, embedding_matrix, device: torch.device):
    with torch.no_grad():
        model.embedding.weight.copy_(torch.tensor(embedding_matrix, dtype=torch.float32, device=device))

