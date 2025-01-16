import torch
from torch import nn
from torch.nn import functional as F

class MemoryBank:
    def __init__(self, feature_dim, max_size=100):
        """
        Memory bank to store features.

        Args:
          feature_dim (int): Dimension of features to store.
          max_size (int): Maximum number of features to store.
        """
        self.max_size = max_size
        self.features = torch.empty((0, feature_dim), dtype=torch.float)

    def add(self, features):
        """Add new features to the memory bank."""
        self.features = torch.cat([self.features, features], dim=0)
        if self.features.size(0) > self.max_size:
            self.features = self.features[-self.max_size:]

    def retrieve(self, query, top_k=5):
        """
        Retrieve the most similar features to the query.

        Args:
          query (torch.Tensor): Query features of shape (N, D).
          top_k (int): Number of nearest neighbors to retrieve.

        Returns:
          (torch.Tensor): Retrieved features of shape (top_k, D).
        """
        similarities = torch.mm(query, self.features.T)
        top_k_indices = torch.topk(similarities, k=top_k, dim=-1).indices
        return self.features[top_k_indices]
