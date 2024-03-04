import torch.nn as nn
import torch.nn.functional as f

class DQN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(DQN, self).__init__()
        self.convLayer1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.convLayer2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.convLayer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.lineLayer1 = nn.Linear(64 * 4 * 4, 512)
        self.outpLayer1 = nn.Linear(512, output_size)

    def forward(self, x):
        x = f.relu(self.convLayer1(x))
        x = f.relu(self.convLayer2(x))
        x = f.relu(self.convLayer3(x))
        x = x.view(x.size(0), -1)
        x = f.relu(self.lineLayer1(x))
        return self.outpLayer1(x)