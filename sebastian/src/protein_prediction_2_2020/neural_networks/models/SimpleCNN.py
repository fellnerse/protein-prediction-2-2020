import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn

from protein_prediction_2_2020.neural_networks.models.PLModel import PLModel


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


class LightAttention(PLModel):
    def __init__(
        self,
        embeddings_dim=1024,
        output_dim=11,
        dropout=0.25,
        kernel_size=9,
        conv_dropout: float = 0.25,
    ):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(
            embeddings_dim,
            embeddings_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.attention_convolution = nn.Conv1d(
            embeddings_dim,
            embeddings_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            # nn.BatchNorm1d(32),
        )

        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attention_convolution(
            x
        )  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        # attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        o1 = torch.sum(
            o * self.softmax(attention), dim=-1
        )  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]


class BetterAttention(PLModel):
    def __init__(self, embeddings_dim=1024, kernel_size=32):
        super(BetterAttention, self).__init__()
        self.attention_convolution = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size, stride=1, padding=kernel_size // 2),  # -> 1024
            nn.MaxPool1d(2),  # -> 512
            nn.LeakyReLU(),
            nn.Conv1d(1, 1, kernel_size, stride=1, padding=kernel_size // 2),  # -> 512
            nn.MaxPool1d(2),  # -> 256
            nn.LeakyReLU(),
            nn.Conv1d(1, 1, kernel_size, stride=1, padding=kernel_size // 2),  # -> 256
            nn.MaxPool1d(2),  # -> 128
            nn.LeakyReLU(),
            nn.Conv1d(1, 1, kernel_size, stride=1, padding=kernel_size // 2),  # -> 128
            nn.MaxPool1d(2),  # -> 64
            nn.LeakyReLU(),
            nn.Conv1d(
                1, 1, kernel_size // 2, stride=1, padding=kernel_size // 4
            ),  # -> 64
            nn.MaxPool1d(2),  # -> 32
            nn.LeakyReLU(),
            nn.Conv1d(
                1, 1, kernel_size // 4, stride=1, padding=kernel_size // 8
            ),  # -> 32
            nn.MaxPool1d(2),  # -> 16
            nn.LeakyReLU(),
            nn.Conv1d(
                1, 1, kernel_size // 8, stride=1, padding=kernel_size // 16
            ),  # -> 8
            nn.MaxPool1d(2),  # -> 4
            nn.LeakyReLU(),
            nn.Conv1d(
                1, 1, kernel_size // 16, stride=1, padding=kernel_size // 32
            ),  # -> 4
            nn.MaxPool1d(2),  # -> 2
            nn.LeakyReLU(),
            nn.Conv1d(1, 1, kernel_size // 32, stride=1, padding=0),  # -> 3
            nn.LeakyReLU(),
            nn.Conv1d(1, 1, 4, stride=1, padding=0),  # -> 1
        )
        self.feature_convolution = nn.Identity()

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 32),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(32),
        )

        self.output = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attention_convolution(
            x.permute((2, 0, 1))
        )  # [sequence_length, 1, 1]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        # attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        o1 = torch.sum(
            o * F.softmax(attention.permute(2, 1, 0)), dim=-1
        )  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]


class ImageModel(PLModel):
    def __init__(self):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor):
        return self.resnet18(x.unsqueeze(1).repeat(1, 3, 1, 1))
