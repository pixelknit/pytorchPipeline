import torch
import torch.nn as nn

class TinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
                    nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )

        self.conv_block_2 = nn.Sequential(
                    nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                )
        
        self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(in_features=hidden_units * 13 * 13, out_features=output_shape)
                )

    def forward(self,x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)

        return x
