import logging
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pandas as pd
import torch
from Bio.SeqIO.FastaIO import SimpleFastaParser
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

logger = logging.getLogger(__name__)


class ProteinDataset(Dataset):
    data_splits = ["test", "train", "val"]

    annotation_file_name = "merged_anno.txt"

    aug_annotation_file_name = "augmented_anno.txt"

    @staticmethod
    def data_split_to_file_name(data_split: str, fasta_file: bool = True):
        return (
            f"ec_vs_NOec_pide20_c50_{data_split}.{ 'fasta'  if  fasta_file else 'h5'}"
        )

    def __init__(self, data_folder: str, data_split: str = "train", augmentation: bool = False):

        if data_split not in self.data_splits:
            raise ValueError(f"{data_split} is not a valid data split name.")

        self.root = Path(data_folder)

        if not self.root.exists():
            raise ValueError(f"{self.root.resolve()} does not exist.")

        self.data_path = self.root / self.data_split_to_file_name(
            data_split, fasta_file=False
        )

        self.sequence_path = self.root / self.data_split_to_file_name(
            data_split, fasta_file=True
        )

        with h5py.File(self.data_path, "r") as f:
            self.key_map = dict()
            self.keys = list(f.keys())
            for i, key in enumerate(self.keys):
                self.key_map[i] = key

        with open(self.sequence_path, "r") as fasta:
            ids = []
            seqs = []
            lengths = []
            for title, seq in SimpleFastaParser(fasta):
                ids.append(title)
                seqs.append(seq)
                lengths.append(len(seq))
            data = {"sequence": seqs, "length": lengths}
            self.data_frame = pd.DataFrame(data=data, index=ids)

        self.annotations = pd.read_csv(
            self.root / self.annotation_file_name,
            sep="\\t",
            names=["index", "EC"],
            index_col=0,
            engine="python",
        )

        self.data_frame = self.data_frame.merge(
            self.annotations, how="left", left_index=True, right_index=True
        )

        if data_split == "train" and augmentation:
            aug_annotations = pd.read_csv(
                self.root / self.aug_annotation_file_name,
                sep="\\t",
                names=["index", "EC"],
                index_col=0,
                engine="python",
            )
            self.data_frame = pd.concat([self.data_frame, aug_annotations], verify_integrity=True)

        self.data_frame["ec_or_nc"] = np.where(self.data_frame["EC"].isna(), "NC", "EC")
        self.data_frame["EC"] = self.data_frame["EC"].fillna("0")
        self.data_frame["EC_first"] = self.data_frame["EC"].str[0]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.key_map[index]
        with h5py.File(self.data_path, "r") as f:
            embeddings = torch.tensor(f[key][:], dtype=torch.float32).T

        label = torch.tensor(
            [self.class_to_idx(self.data_frame["EC_first"][key])], dtype=torch.long
        )

        return embeddings, label

    @staticmethod
    def class_to_idx(label: str) -> float:
        return int(label)


class BalancedSampler(WeightedRandomSampler):
    def __init__(
        self, dataset: ProteinDataset, replacement=True, predefined_weights=None
    ):
        class_weight_dict = self._calculate_class_weights(
            dataset, predefined_weights=predefined_weights
        )
        class_weights = np.zeros((8,))
        for key, value in class_weight_dict.items():
            class_weights[dataset.class_to_idx(key)] = value

        logger.warning(f"Classweights, that are used for {dataset}: {class_weights}")

        # this is in wrong order for now
        targets = np.zeros((len(dataset)), dtype=np.int)
        for i in range(len(dataset)):
            key = dataset.key_map[i]
            targets[i] = dataset.class_to_idx(dataset.data_frame["EC_first"][key])

        weights = class_weights[targets]

        super().__init__(weights, num_samples=len(dataset), replacement=replacement)

    @staticmethod
    def _calculate_class_weights(
        dataset: ProteinDataset, predefined_weights=None
    ) -> Dict[str, float]:
        labels, counts = np.unique(
            np.array(dataset.data_frame["EC_first"]), return_counts=True
        )
        if predefined_weights is not None:
            counts = predefined_weights / counts
        else:
            counts = 1 / counts
        counts /= counts.sum()

        weight_dict = {
            class_idx: 0 for class_idx in set(dataset.data_frame["EC_first"])
        }

        for label, count in zip(labels, counts):
            weight_dict[label] = count

        return weight_dict


def collate_fn(data):
    ret_x = []
    ret_y = []

    for datapoint in data:
        x, y = datapoint
        ret_x.append(x)
        ret_y.append(y)

    ret_y = torch.stack(ret_y, dim=0)

    return ret_x, ret_y
