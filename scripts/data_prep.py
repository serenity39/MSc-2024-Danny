"""Prepare data for pre-training.

Format the MSMARCO dataset for pre-training BERT.
"""

import os

import pandas as pd

# Define paths
raw_dir = "../data/msmarco/raw"
preprocessed_dir = "../data/msmarco/preprocessed"

# Make sure the preprocessed directory exists
os.makedirs(preprocessed_dir, exist_ok=True)

# Load the queries and top passages
queries_df = pd.read_csv(
    os.path.join(raw_dir, "passv2_dev_queries.tsv"),
    sep="\t",
    header=None,
    names=["qid", "query"],
)
top_passages_df = pd.read_csv(
    os.path.join(raw_dir, "dev_top100.txt"),
    sep="\t",
    header=None,
    names=["qid", "pid", "passage"],
)

# Merge the queries with their corresponding passages
merged_df = queries_df.merge(top_passages_df, on="qid")

# Each line in the new file is a concatenation of the query and a passage
with open(
    os.path.join(preprocessed_dir, "msmarco_dev_data.txt"),
    "w",
    encoding="utf-8",
) as file:
    for index, row in merged_df.iterrows():
        # Concatenate the query and the passage with a special token in between
        line = row["query"] + " [SEP] " + row["passage"] + "\n"
        file.write(line)

print("Data preparation complete.")
