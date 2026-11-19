"""I20 CNN architecture adapted from the authors' Model/cnn_model.py.

Source: trend_submit/trend_code_submit/Model/cnn_model.py, CNNModel class;
settings: Misc/config.py, EMP_CNN_BL_SETTING[20].
This module contains no data processing or filesystem side effects.
The state_dict layout follows the original three-block CNNModel.
"""

import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def init_weights(module):
    if type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
    elif type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)


class CNN20(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        previous_channels = 1
        for layer, channels in enumerate((64, 128, 256)):
            blocks.append(nn.Sequential(
                nn.Conv2d(
                    previous_channels, channels, kernel_size=(5, 3),
                    stride=(3, 1) if layer == 0 else (1, 1),
                    padding=(2, 1),
                    dilation=(2, 1) if layer == 0 else (1, 1),
                ),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(),
                nn.MaxPool2d((2, 1), ceil_mode=True),
            ))
            previous_channels = channels
        self.conv_layers = nn.Sequential(*blocks, Flatten(), nn.Dropout(0.5))
        # Match the original constructor, including its training-mode dummy pass.
        with torch.no_grad():
            feature_count = self.conv_layers(torch.rand(1, 1, 64, 60)).shape[1]
        if feature_count != 46080:
            raise RuntimeError(f"Unexpected flattened feature count: {feature_count}")
        self.fc = nn.Linear(feature_count, 2)
        self.conv_layers.apply(init_weights)
        self.fc.apply(init_weights)

    def forward(self, x):
        return self.fc(self.conv_layers(x))
