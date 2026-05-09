import torch
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, n_actions):
        super().__init__()

        # Architecture from 2013 DQN paper (arXiv version)
        self.conv = nn.Sequential(
            nn.Conv2d(4,  16, kernel_size=8, stride=4), nn.ReLU(),  # 16 filters (not 32)
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),  # 32 filters (not 64)
            # No third conv layer in 2013 paper
        )

        # Conv output: 9×9×32 = 2592
        self.fc = nn.Sequential(
            nn.Linear(2592, 256), nn.ReLU(),  # 256 units (not 512)
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)
