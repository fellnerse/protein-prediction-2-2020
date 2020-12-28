import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn

from protein_prediction_2_2020.neural_networks.models.PLModel import PLModel


class SimpleLSTM(PLModel):
    def __init__(self, num_layer):
        super(SimpleLSTM, self).__init__()
        self.num_layer = num_layer
        self.lstm = nn.LSTM(1024, 128, num_layer, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1)
        _, (x, _) = self.lstm(x)
        x = x.view(self.num_layer, 2, batch_size, 128)
        x = x.permute(2, 0, 1, 3)
        x = x.reshape(batch_size, self.num_layer, 256)
        x = x[:, self.num_layer - 1, :]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
