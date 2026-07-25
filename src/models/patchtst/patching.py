import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):

    def __init__(self, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len  = patch_len
        self.stride     = stride
        self.projection = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        embeddings = self.projection(patches)

        return embeddings
