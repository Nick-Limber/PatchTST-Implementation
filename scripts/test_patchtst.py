import torch
from src.models.patchtst.patching import PatchEmbedding
from src.models.patchtst.encoder import PatchTSTEncoder
from src.models.patchtst.model import PatchTST

CONTEXT_LEN   = 96
PATCH_LEN     = 16
STRIDE        = 8
D_MODEL       = 64
N_HEADS       = 4
N_LAYERS      = 3
D_FEEDFORWARD = 256
DROPOUT       = 0.1
HORIZON       = 28
BATCH         = 32

print("Testing PatchEmbedding...")
patch_emb = PatchEmbedding(PATCH_LEN, STRIDE, D_MODEL)
x = torch.randn(BATCH, CONTEXT_LEN)
patches = patch_emb(x)
n_patches = (CONTEXT_LEN - PATCH_LEN) // STRIDE + 1
assert patches.shape == (BATCH, n_patches, D_MODEL)
print(f"  OK: {list(x.shape)} → {list(patches.shape)}")

print("Testing PatchTSTEncoder...")
encoder = PatchTSTEncoder(D_MODEL, N_HEADS, N_LAYERS, D_FEEDFORWARD, DROPOUT, n_patches)
encoded = encoder(patches)
assert encoded.shape == (BATCH, n_patches, D_MODEL)
print(f"  OK: {list(patches.shape)} → {list(encoded.shape)}")

print("Testing full PatchTST...")
model = PatchTST(
    context_len=CONTEXT_LEN, patch_len=PATCH_LEN, stride=STRIDE,
    d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
    d_feedforward=D_FEEDFORWARD, dropout=DROPOUT, horizon=HORIZON,
)
x = torch.randn(BATCH, CONTEXT_LEN)
forecast = model(x)
assert forecast.shape == (BATCH, HORIZON)
print(f"  OK: {list(x.shape)} → {list(forecast.shape)}")

print("Testing gradient flow...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
target = torch.randn(BATCH, HORIZON)
optimizer.zero_grad()
loss = ((model(x) - target) ** 2).mean()
loss.backward()
optimizer.step()
grad = model.patch_embedding.projection.weight.grad
assert grad is not None
assert not torch.isnan(grad).any()
print(f"  OK: loss={loss.item():.4f}, grad_norm={grad.norm().item():.4f}")

total = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total:,}")
print("\nAll checks passed.")
