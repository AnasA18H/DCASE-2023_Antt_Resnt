import torch
from networks.dcase2023t2_ae.network import AENet

model = AENet(input_dim=640, block_size=64)
x = torch.randn(32, 640)
out, z = model(x)
print("Forward pass successful!")
print("Output shape:", out.shape)
print("Z shape:", z.shape)
