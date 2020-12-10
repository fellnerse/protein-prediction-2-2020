import click
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from protein_prediction_2_2020.neural_networks.models.SimpleCNN import SimpleCNN
from protein_prediction_2_2020.neural_networks.utils.dataset import ProteinDataset


@click.command()
@click.option(
    "--data_folder", type=click.Path(exists=True), help="Path to folder containing data"
)
@click.option(
    "--log_folder",
    type=click.Path(exists=True),
    help="Path to folder used for PL logging",
)
def train(data_folder, log_folder):
    trainset = ProteinDataset(data_folder=data_folder, data_split="train")
    valset = ProteinDataset(data_folder=data_folder, data_split="val")

    model = SimpleCNN()
    trainer = pl.Trainer(gpus=-1, default_root_dir=log_folder, val_check_interval=0.02)
    trainer.fit(
        model, DataLoader(trainset, num_workers=8), DataLoader(valset, num_workers=8)
    )


if __name__ == "__main__":
    train()
