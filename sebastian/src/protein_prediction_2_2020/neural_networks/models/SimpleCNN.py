import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class SimpleCNN(pl.LightningModule):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1024, 32, 7)
        self.conv2 = nn.Conv1d(32, 1, 7)
        self.pool = nn.AdaptiveMaxPool1d(256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _step(self, batch, batch_idx, name: str = "train"):
        x, y = batch
        pred = self.forward(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        self.log(name + "_Loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer