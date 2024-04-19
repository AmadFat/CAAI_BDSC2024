import torch
import torch.nn as nn


class predictor(nn.Module):
    def __init__(self, voc_size, embedding_dim, half_window_size):
        super().__init__()
        self.half_window_size = half_window_size
        self.embedding = nn.Embedding(num_embeddings=voc_size, embedding_dim=embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.activation = nn.SELU()
        self.dropout = nn.Dropout(0.5)
        self.ln2 = nn.LayerNorm(2 * embedding_dim)
        self.fc2 = nn.Linear(2 * embedding_dim, 2 * half_window_size * voc_size)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x.long())
        x = self.ln1(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.ln2(x)
        x = self.fc2(x)
        return x.view(x.shape[0], 2 * self.half_window_size, -1)
