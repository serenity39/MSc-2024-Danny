"""Evaluate model using Pyserini and TREC Eval."""

from datasets import load_from_disk

dataset = load_from_disk("../data/hf_datasets/depth_based_50_50")

print(dataset.select(range(5)))
