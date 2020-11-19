# %%
from pathlib import Path

import pandas as pd
from Bio.SeqIO.FastaIO import SimpleFastaParser

root = Path("sebastian/data/")
annotation_file = "merged_anno.txt"

seq_files = {
    "test": "ec_vs_NOec_pide20_c50_test.fasta",
    "train": "ec_vs_NOec_pide20_c50_train.fasta",
    "val": "ec_vs_NOec_pide20_c50_val.fasta",
}

data_frames = {}

for data_set_name, file_name in seq_files.items():
    with open(root / file_name) as fasta_file:
        ids = []
        seqs = []
        lengths = []
        for title, seq in SimpleFastaParser(fasta_file):
            ids.append(title)
            seqs.append(seq)
            lengths.append(len(seq))
        data = {"sequence": seqs, "length": lengths}
        data_frames[data_set_name] = pd.DataFrame(data=data, index=ids)

annotations = pd.read_csv(
    root / annotation_file,
    sep="\\t",
    names=["index", "EC"],
    index_col=0,
    engine="python",
)

for data_set_name in seq_files.keys():
    data_frames[data_set_name] = data_frames[data_set_name].merge(
        annotations, how="left", left_index=True, right_index=True
    )

#%%
non_enzyms = {}
enzyms = {}
for data_set_name in seq_files.keys():
    df = data_frames[data_set_name]
    non_enzyms[data_set_name] = df.loc[df["EC"].isna()]
    enzyms[data_set_name] = df.loc[df["EC"].notna()]
    print("{}:\t#enzyms\t\t{}".format(data_set_name, len(enzyms[data_set_name])))
    print("{}:\t#non-enzyms\t{}".format(data_set_name, len(non_enzyms[data_set_name])))

for data_set_name in seq_files.keys():
    df = data_frames[data_set_name]["length"]
    df.plot.kde(title="Length Distributions", label=data_set_name + "_all", legend=True)
    df_enz = enzyms[data_set_name]["length"]
    df_enz.plot.kde(
        title="Length Distributions", label=data_set_name + "_enz", legend=True
    )
    df_nonenz = non_enzyms[data_set_name]["length"]
    df_nonenz.plot.kde(
        title="Length Distributions", label=data_set_name + "_nonenz", legend=True
    )
