"""Evaluate model using Pyserini and TREC Eval."""

import random

import pandas as pd
from datasets import ClassLabel, Sequence, load_from_disk


def show_random_elements(dataset, num_examples=10):
    """Show random elements from a Huggingface dataset.

    Args:
        dataset: dataset to pick the examples from.
        num_examples (optional): number of examples to show. Defaults to 10.
    """
    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(
                lambda x: [typ.feature.names[i] for i in x]
            )
    print(df)


def show_first_elements(dataset, num_examples=100):
    """Show the first elements from a Huggingface dataset.

    Args:
        dataset: dataset to pick the examples from.
        num_examples (optional): number of examples to show. Defaults to 50.
    """
    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."

    df = pd.DataFrame(dataset.select(range(num_examples)))
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(
                lambda x: [typ.feature.names[i] for i in x]
            )
    print(df)


dataset = load_from_disk("../data/hf_datasets/depth_based_50_50")

# show_random_elements(dataset)
# print()
show_first_elements(dataset)
