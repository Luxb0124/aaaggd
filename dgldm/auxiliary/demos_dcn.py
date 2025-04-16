import torch
from torchvision.ops import DeformConv2d


dcn = DeformConv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, dilation=1,)
data = torch.randn(size=(8, 64, 48, 48))
out = dcn(data)
print(out.shape)
