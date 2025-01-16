import torch
from torch import nn
from torch.nn import functional as F

class MemoryAttention(nn.Module):
    def __init__(self, feature_dim):
        """
        Memory attention module.

        Args:
          feature_dim (int): Dimension of the feature embeddings.
        """
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, memory_features):
        """
        Forward pass for memory attention.

        Args:
          query (torch.Tensor): Query features of shape (N, D).
          memory_features (torch.Tensor): Memory bank features of shape (M, D).

        Returns:
          torch.Tensor: Attention-weighted features of shape (N, D).
        """
        query = self.query_proj(query)
        keys = self.key_proj(memory_features)
        values = self.value_proj(memory_features)

        attention_weights = self.softmax(torch.mm(query, keys.T))
        context = torch.mm(attention_weights, values)
        return context
