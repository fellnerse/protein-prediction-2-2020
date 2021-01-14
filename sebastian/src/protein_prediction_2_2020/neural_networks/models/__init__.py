import pytorch_lightning as pl
import torch
from torch.nn import functional as F


class PLModel(pl.LightningModule):
    def _step(self, batch, batch_idx, name: str = "train"):
        x, y = batch
        pred = torch.zeros_like(y)

        for idx, x_ in enumerate(x):
            pred[idx] = self.forward(x_.unsqueeze(0)).squeeze(0)

        loss = F.binary_cross_entropy_with_logits(pred, y)
        self.log(name + "_loss", loss)
        # self.logger.experiment.add_scalars(
        #     "loss", {name: loss}, global_step=self.global_step
        # )

        pred = torch.sigmoid(pred.detach())
        pred = torch.round(pred.data)
        correct = (pred == y).sum().item()

        self.log(name + "_acc", correct / pred.size(0))
        # self.logger.experiment.add_scalars(
        #     "acc", {name: correct / pred.size(0)}, global_step=self.global_step
        # )

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
