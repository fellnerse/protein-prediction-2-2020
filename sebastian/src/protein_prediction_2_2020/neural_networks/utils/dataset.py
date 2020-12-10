from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from Bio.SeqIO.FastaIO import SimpleFastaParser
from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    data_splits = ["test", "train", "val"]

    annotation_file_name = "merged_anno.txt"

    @staticmethod
    def data_split_to_file_name(data_split: str, fasta_file: bool = True):
        return (
            f"ec_vs_NOec_pide20_c50_{data_split}.{ 'fasta'  if  fasta_file else 'h5'}"
        )

    def __init__(self, data_folder: str, data_split: str = "train"):

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
        self.data_frame["ec_or_nc"] = np.where(self.data_frame["EC"].isna(), "NC", "EC")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.key_map[index]
        with h5py.File(self.data_path, "r") as f:
            embeddings = torch.tensor(f[key][:], dtype=torch.float32).T
        if self.data_frame["ec_or_nc"][key] in "EC":
            bin_label = torch.Tensor([1])
        else:
            bin_label = torch.Tensor([0])
        return [embeddings, bin_label]
