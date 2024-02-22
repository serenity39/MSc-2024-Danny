"""Prepare data for pre-training.

Format the MSMARCO dataset for pre-training BERT.
"""

import json
import os

import pandas as pd

# Define paths
raw_dir = "../data/msmarco/raw"
preprocessed_dir = "../data/msmarco/preprocessed"
corpus_dir = "../data/msmarco/raw/corpus/msmarco_v2_passage"

# Make sure the preprocessed directory exists
os.makedirs(preprocessed_dir, exist_ok=True)

# Load the queries
queries_df = pd.read_csv(
    os.path.join(raw_dir, "passv2_dev_queries.tsv"),
    sep="\t",
    dtype={"qid": str},
    header=None,
    names=["qid", "query"],
)

# Load the top passages (which contain the passage IDs, not the passage text)
top_passages_df = pd.read_csv(
    os.path.join(raw_dir, "passv2_dev_top100.txt"),
    delim_whitespace=True,
    dtype={"qid": str},
    header=None,
    names=["qid", "Q0", "docid", "rank", "score", "runstring"],
    usecols=["qid", "docid"],
)

# Extract the passage IDs that we need to retrieve from the corpus
needed_passage_ids = set(top_passages_df["docid"].tolist())

# Prepare a dictionary to hold the passage ID to text mapping
passage_id_to_text = {}

print("Loading passage texts...")

# Loop through the corpus directory and read each JSONL file
for filename in os.listdir(corpus_dir):
    file_path = os.path.join(corpus_dir, filename)
    if filename.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                pid = data["pid"]
                if pid in needed_passage_ids:
                    passage_id_to_text[pid] = data["passage"]


# Convert the passage ID to text mapping into a DataFrame
passages_df = pd.DataFrame(
    list(passage_id_to_text.items()), columns=["docid", "passage_text"]
)

# Merge the queries with their corresponding passages
merged_queries_top_passages = pd.merge(queries_df, top_passages_df, on="qid")
merged_df = pd.merge(merged_queries_top_passages, passages_df, on="docid")

# Write the merged data to a file suitable for pre-training
with open(
    os.path.join(preprocessed_dir, "msmarco_dev_data.txt"),
    "w",
    encoding="utf-8",
) as file:
    for index, row in merged_df.iterrows():
        line = row["query"] + " [SEP] " + row["passage_text"] + "\n"
        file.write(line)

print("Data preparation complete.")
print(
    "Output file created at: ,",
    os.path.join(preprocessed_dir, "msmarco_dev_data.txt"),
)
