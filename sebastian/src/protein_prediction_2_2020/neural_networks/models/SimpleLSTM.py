import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn

from protein_prediction_2_2020.neural_networks.models.PLModel import PLModel


class SimpleLSTM(PLModel):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(1024, 256, 1, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        _, (x, _) = self.lstm(x)
        x = x.permute(1, 0, 2)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _step(self, batch, batch_idx, name: str = "train"):
        x, y = batch
        pred = torch.zeros_like(y)

        self._fill_preds(pred, x)

        loss = F.binary_cross_entropy_with_logits(pred, y)
        self.log(name + "_loss", loss.item())
        # self.logger.experiment.add_scalars(
        #     "loss", {name: loss}, global_step=self.global_step
        # )

        pred = torch.sigmoid(pred.detach())
        pred = torch.round(pred.data)
        correct = (pred == y).sum().item()

        self.log(name + "_acc", (correct / pred.size(0)))
        # self.logger.experiment.add_scalars(
        #     "acc", {name: correct / pred.size(0)}, global_step=self.global_step
        # )

        return loss

    def _fill_preds(self, pred, x):
        for idx, x_ in enumerate(x):
            pred[idx] = self.forward(x_.unsqueeze(0)).squeeze(0)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
