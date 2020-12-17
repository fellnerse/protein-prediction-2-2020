from pathlib import Path

import click
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from protein_prediction_2_2020.neural_networks.models.SimpleCNN import BetterAttention
from protein_prediction_2_2020.neural_networks.models.SimpleCNN import ComplexCNN
from protein_prediction_2_2020.neural_networks.models.SimpleCNN import ImageModel
from protein_prediction_2_2020.neural_networks.models.SimpleCNN import LightAttention
from protein_prediction_2_2020.neural_networks.models.SimpleCNN import SimpleCNN
from protein_prediction_2_2020.neural_networks.utils.dataset import BalancedSampler
from protein_prediction_2_2020.neural_networks.utils.dataset import collate_fn
from protein_prediction_2_2020.neural_networks.utils.dataset import ProteinDataset


@click.command()
@click.option(
    "--data_folder", type=click.Path(exists=True), help="Path to folder containing data"
)
@click.option(
    "--log_folder",
    type=click.Path(exists=True),
    help="Path to folder used for PL logging.",
)
@click.option(
    "--validate_after", default=0.1, help="Validate after this train data percentage."
)
@click.option(
    "--num_epochs", default=5, help="The number of epochs the model will be trained."
)
@click.option("--batch_size", default=64, help="Batch size used with dataloader.")
@click.option("--run_name", prompt="Name of run")
def train(data_folder, log_folder, validate_after, num_epochs, batch_size, run_name):
    trainset = ProteinDataset(data_folder=data_folder, data_split="train")
    valset = ProteinDataset(data_folder=data_folder, data_split="val")
    testset = ProteinDataset(data_folder=data_folder, data_split="test")

    # model = ComplexCNN()
    # model = SimpleCNN()
    # model = LightAttention(output_dim=1)
    # model = BetterAttention()
    model = ImageModel()
    trainer = pl.Trainer(
        gpus=-1,
        default_root_dir=Path(log_folder) / run_name,
        val_check_interval=validate_after,
        max_epochs=num_epochs,
    )
    trainer.fit(
        model,
        DataLoader(
            trainset,
            num_workers=8,
            batch_size=batch_size,
            collate_fn=collate_fn,
            # shuffle=True
            sampler=BalancedSampler(trainset),
        ),
        DataLoader(valset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn),
    )

    res = trainer.test(model, DataLoader(testset, num_workers=8))
    print(res)


if __name__ == "__main__":
    train()
