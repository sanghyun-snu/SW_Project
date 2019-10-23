from vgg import vgg16_gap
from resnet import resnet50
import torch

model = vgg16_gap(pretrained=True, num_classes=20)

x = torch.zeros((3, 3, 321, 321))

y = model(x)
print(y.shape)