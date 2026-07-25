import torch
import torch.nn as nn


class PatchTSTEncoder(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_feedforward: int,
        dropout: float,
        max_patches: int,
    ):
        super().__init__()

        self.pos_embedding = nn.Embedding(max_patches, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

       
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, n_patches, d_model = x.shape

        positions = torch.arange(n_patches, device=x.device)

        x = x + self.pos_embedding(positions)
        x = self.encoder(x)

        return x
