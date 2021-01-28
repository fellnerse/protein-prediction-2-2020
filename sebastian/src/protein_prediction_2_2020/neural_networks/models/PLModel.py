import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics import ConfusionMatrix


class PLModel(pl.LightningModule):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.confmat = ConfusionMatrix(num_classes=output_dim)

    def _step(self, batch, batch_idx, name: str = "train"):
        x, y = batch
        y = y.squeeze(1)
        pred = torch.zeros(y.shape[0], self.output_dim).cuda()

        for idx, x_ in enumerate(x):
            pred[idx] = self.forward(x_.unsqueeze(0)).squeeze(0)

        loss = F.cross_entropy(pred, y)
        self.log(name + "_loss", loss)

        pred = torch.argmax(pred.detach(), dim=1)
        correct = (pred == y).sum().item()

        if name == "test":
            self.confmat.update(pred, y)

        self.log(name + "_acc", correct / pred.size(0))

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="test")

    def on_test_epoch_end(self) -> None:
        import pandas as pd
        import numpy as np
        import seaborn as sn
        import matplotlib.pyplot as plt
        from torchvision import transforms
        from PIL import Image
        import io

        tb = self.logger.experiment
        conf_mat = self.confmat.compute().detach().cpu().numpy().astype(np.int)
        df_cm = pd.DataFrame(
            conf_mat,
            index=np.arange(self.output_dim),
            columns=np.arange(self.output_dim))
        plt.figure()
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')
        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        im = Image.open(buf)
        im = transforms.ToTensor()(im)
        tb.add_image("test_confusion_matrix", im, global_step=self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
