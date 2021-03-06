import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models as models

from protein_prediction_2_2020.neural_networks.models import PLModel


class SimpleCNN(PLModel):
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


class ComplexCNN(PLModel):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv1d(1024, 128, 7)
        self.conv1_bn = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, 7)
        self.conv2_bn = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 32, 7)
        self.conv3_bn = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, 1, 7)
        self.conv4_bn = nn.BatchNorm1d(1)

        self.pool = nn.AdaptiveMaxPool1d(256)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = self.pool(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ImageModel(PLModel):
    def __init__(self):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor):
        return self.resnet18(x.unsqueeze(1).repeat(1, 3, 1, 1))
