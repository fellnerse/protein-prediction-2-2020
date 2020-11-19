# %% Run mmseqs and generate results
import subprocess

subprocess.run(
    "mmseqs easy-search "
    "sebastian/data/ec_vs_NOec_pide20_c50_test.fasta  "
    "sebastian/data/ec_vs_NOec_pide20_c50_train.fasta "
    "sebastian/src/protein-prediction-2-2020/mmseq_baseline/result_test.m8 "
    "sebastian/src/protein-prediction-2-2020/mmseq_baseline/tmp",
    shell=True,
    check=True,
)

subprocess.run(
    "mmseqs easy-search "
    "sebastian/data/ec_vs_NOec_pide20_c50_val.fasta  "
    "sebastian/data/ec_vs_NOec_pide20_c50_train.fasta "
    "sebastian/src/protein-prediction-2-2020/mmseq_baseline/result_val.m8 "
    "sebastian/src/protein-prediction-2-2020/mmseq_baseline/tmp",
    shell=True,
    check=True,
)

#%% load one result file and annotations
import pandas as pd
from pathlib import Path

results = pd.read_csv(
    "sebastian/src/protein-prediction-2-2020/mmseq_baseline/result_test.m8",
    sep="\\t",
    header=None,
    engine="python",
)

root = Path("sebastian/data/")
annotation_file = "merged_anno.txt"

annotations = pd.read_csv(
    root / annotation_file,
    sep="\\t",
    names=["index", "EC"],
    index_col=0,
    engine="python",
)

#%% define functions to convert name of EC to label
def seq_to_binary_label(seq: str):
    try:
        annotations["EC"][seq]
        return 1.0
    except KeyError:
        return 0.0


def seq_to_mc_label(seq: str):
    try:
        return float(annotations["EC"][seq][0])
    except KeyError:
        return 0.0


# %% calculate predictions
import torch

binary_preds = torch.zeros((results.shape[0], 2), dtype=torch.int)
mc_preds = torch.zeros((results.shape[0], 2), dtype=torch.int)

numpy_results = results[[0, 1]].to_numpy()

for idx, (query, target) in enumerate(numpy_results):
    binary_preds[idx] = torch.tensor(
        [seq_to_binary_label(query), seq_to_binary_label(target)]
    )
    mc_preds[idx] = torch.tensor([seq_to_mc_label(query), seq_to_mc_label(target)])

# %% print accuracy

print(
    f"Binary acc: {(binary_preds[:, 0]==binary_preds[:, 1]).sum().data.numpy() / binary_preds.shape[0]}"
)

print(
    f"Multi class acc: {(mc_preds[:, 0]==mc_preds[:, 1]).sum().data.numpy() / mc_preds.shape[0]}"
)
