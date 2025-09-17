import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEQCNN(nn.Module):
    """
    Simple CNN for mel-spectrogram regression (e.g., predicting EQ gains).
    Input: (batch, 1, mel_bins, time_frames)
    Output: (batch, N_bands)
    """
    def __init__(self, mel_bins=64, time_frames=50, n_bands=6):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
        self.pool = nn.MaxPool2d((2,2))
        # Compute output shape after 3 conv + 2 pool for flatten
        mel_out = mel_bins // 2 // 2
        time_out = time_frames // 2 // 2
        self.fc1 = nn.Linear(64 * mel_out * time_out, 128)
        self.fc2 = nn.Linear(128, n_bands)

    def forward(self, x):
        # x: (batch, 1, mel_bins, time_frames)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out
