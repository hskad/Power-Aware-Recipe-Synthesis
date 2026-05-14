import torch
from torch import nn


class PowerPredictor(nn.Module):
    def __init__(
        self,
        graph_feature_dim: int,
        recipe_vocab_size: int,
        hidden_dim: int = 128,
        recipe_emb_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.recipe_emb = nn.Embedding(recipe_vocab_size, recipe_emb_dim)

        self.graph_encoder = nn.Sequential(
            nn.Linear(graph_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + recipe_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_features: torch.Tensor, recipe_idx: torch.Tensor) -> torch.Tensor:
        graph_latent = self.graph_encoder(graph_features)
        recipe_latent = self.recipe_emb(recipe_idx)
        joint = torch.cat([graph_latent, recipe_latent], dim=1)
        return self.head(joint)
