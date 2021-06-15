from torchvision.models import resnet50
import torch
model = resnet50()
input = torch.randn(1, 3, 224, 224)
model_complexity.get_summary(model, (3, 224, 224))

"""
from thop import profile
macs, params = profile(model, inputs=(input, ))
print(macs, params)
"""
