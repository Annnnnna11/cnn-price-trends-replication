"""Same I20 block rule as the original model, varying only block count."""
import torch
from torch import nn
from cnn20 import Flatten, init_weights
from common import architecture_spec

class DepthCNN(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        spec = architecture_spec(depth)
        blocks = []
        previous = 1
        for layer, number in enumerate(spec["channels"]):
            blocks.append(nn.Sequential(
                nn.Conv2d(previous, number, (5, 3),
                          stride=(3, 1) if layer == 0 else (1, 1),
                          padding=(2, 1), dilation=(2, 1) if layer == 0 else (1, 1)),
                nn.BatchNorm2d(number), nn.LeakyReLU(),
                nn.MaxPool2d((2, 1), ceil_mode=True)))
            previous = number
        self.conv_layers = nn.Sequential(*blocks, Flatten(), nn.Dropout(0.5))
        # Preserve the original constructor's dummy forward and initialization order.
        with torch.no_grad():
            flattened = self.conv_layers(torch.rand(1, 1, 64, 60)).shape[1]
        if flattened != spec["flattened_features"]:
            raise RuntimeError("Unexpected feature shape")
        self.fc = nn.Linear(flattened, 2)
        self.conv_layers.apply(init_weights)
        self.fc.apply(init_weights)
        if sum(parameter.numel() for parameter in self.parameters()) != spec["parameters"]:
            raise RuntimeError("Unexpected parameter count")

    def forward(self, images):
        return self.fc(self.conv_layers(images))
