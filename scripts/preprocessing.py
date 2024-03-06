"""Preprocessing script making training sets for BERT."""

import json
import random
from collections import defaultdict

import ir_datasets
from datasets import Dataset
from halo import Halo


# Convert trec-dl-2021 relevance judgments to binary format
def convert_to_binary(trec_qrels):
    """Converts the relevance judgments to binary format.

    Args:
        trec_qrels: The relevance judgments from the TREC-DL-2021 dataset.

    Returns:
        dict: A dictionary containing the query ID, document ID, and
              relevance judgment (0 or 1).
    """
    binary_qrels = []
    for qrel in trec_qrels:
        binary_qrels.append(
            {
                "query_id": qrel.query_id,
                "doc_id": qrel.doc_id,
                "relevance": 1 if qrel.relevance > 0 else 0,
            }
        )
    return binary_qrels


# Negative sampling for train qrels
def negative_sampling(train_qrels, docs, num_neg_samples=1):
    """Sample negative examples for the training set.

    Args:
        train_qrels: The qrels from the msmarco train set.
        docs: The list of document IDs.
        num_neg_samples (optional): Number of negative samples to take for
        each query. Defaults to 1.

    Returns:
        list: A list of negative samples.
    """
    sampled_negatives = []
    for qrel in train_qrels:
        sampled_docs = random.sample(docs, num_neg_samples)
        for doc_id in sampled_docs:
            sampled_negatives.append(
                {"query_id": qrel["query_id"], "doc_id": doc_id, "relevance": 0}
            )
    return sampled_negatives


# Stratify the qrels to create the training sets
def stratify_qrels(  # noqa: C901
    qrels, num_queries, pos_per_query, neg_per_query=0
):
    """Stratify sampling the qrels to create the training sets.

    Args:
        qrels: The relevance judgments.
        num_queries: The number of queries to sample.
        pos_per_query: The number of positive examples to sample per query.
        neg_per_query (optional): The number of negative examples to sample
        per query. Defaults to 0.

    Returns:
        list: A list of stratified samples.
    """
    queries = defaultdict(list)
    for qrel in qrels:
        queries[qrel["query_id"]].append(qrel)

    eligible_queries = []
    for query_id, docs in queries.items():
        pos_count = 0
        neg_count = 0
        for doc in docs:
            if doc["relevance"] == 1:
                pos_count += 1
            elif doc["relevance"] == 0:
                neg_count += 1
        if pos_count >= pos_per_query and neg_count >= neg_per_query:
            eligible_queries.append(query_id)

    selected_queries = random.sample(eligible_queries, num_queries)
    stratified_samples = []
    for query_id in selected_queries:
        pos_samples = []
        neg_samples = []
        for q in queries[query_id]:
            if q["relevance"] == 1:
                pos_samples.append(q)
            elif q["relevance"] == 0:
                neg_samples.append(q)
        pos_selected = random.sample(pos_samples, pos_per_query)
        neg_selected = random.sample(neg_samples, neg_per_query)
        stratified_samples.extend(pos_selected)
        stratified_samples.extend(neg_selected)

    return stratified_samples


def save_samples(samples, file_path):
    """Save the samples to a file.

    Args:
        samples: The samples to save.
        file_path: The path to the file where the samples will be saved.
    """
    with open(file_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    spinner = Halo(text="Loading datasets...", spinner="dots")
    spinner.start()
    # Load datasets
    trec_qrels_dataset = ir_datasets.load(
        "msmarco-passage-v2/trec-dl-2021"
    ).qrels_iter()
    train_qrels_dataset = ir_datasets.load(
        "msmarco-passage-v2/train"
    ).qrels_iter()
    spinner.succeed("Datasets loaded!")

    # Convert the datasets to binary format
    spinner.start("Converting datasets to binary format...")
    trec_qrels_binary = list(convert_to_binary(trec_qrels_dataset))
    spinner.succeed("Datasets converted to binary format.")

    # Create a list of doc IDs for negative sampling
    spinner.start("Creating negative samples...")
    all_doc_ids = set()
    for qrel in train_qrels_dataset:
        all_doc_ids.add(qrel.doc_id)

    train_qrels_binary = []
    for qrel in train_qrels_dataset:
        train_qrels_binary.append(
            {
                "query_id": qrel.query_id,
                "doc_id": qrel.doc_id,
                "relevance": qrel.relevance,
            }
        )

    # Create a set of doc_ids that have been used in train_qrels_binary
    used_doc_ids = set()
    for qrel in train_qrels_binary:
        used_doc_ids.add(qrel["doc_id"])

    # Determine doc_ids for negative sampling
    doc_ids_for_negative_sampling = all_doc_ids - used_doc_ids

    negative_samples = negative_sampling(
        train_qrels_binary, doc_ids_for_negative_sampling
    )
    spinner.succeed("Negative samples created.")

    # Combine positive and negative samples for train qrels
    combined_train_qrels = train_qrels_binary + negative_samples

    # Creating depth-based and shallow-based training sets

    # Depth-based training sets
    spinner.start("Creating depth-based 50/50 training set...")
    depth_50_50 = stratify_qrels(trec_qrels_binary, 50, 50, 0)
    spinner.succeed(
        "Depth-based 50/50 training set created. Size: ", len(depth_50_50)
    )

    spinner.start("Creating depth-based 50/100 training set...")
    depth_50_100 = stratify_qrels(trec_qrels_binary, 50, 100, 0)
    spinner.succeed(
        "Depth-based 50/100 training set created. Size: ", len(depth_50_100)
    )

    # Shallow-based training sets (1:1 ratio)
    spinner.start("Creating shallow-based 2500/1 training set...")
    shallow_2500_1 = stratify_qrels(combined_train_qrels, 2500, 1, 1)
    spinner.succeed(
        "Shallow-based 2500/1 training set created. Size: ", len(shallow_2500_1)
    )

    spinner.start("Creating shallow-based 5000/1 training set...")
    shallow_5000_1 = stratify_qrels(combined_train_qrels, 5000, 1, 1)
    spinner.succeed(
        "Shallow-based 5000/1 training set created. Size: ", len(shallow_5000_1)
    )

    # Convert the training sets to datasets
    spinner.start("Converting training sets to datasets...")
    depth_50_50_dataset = Dataset.from_dict(depth_50_50)
    depth_50_100_dataset = Dataset.from_dict(depth_50_100)
    shallow_2500_1_dataset = Dataset.from_dict(shallow_2500_1)
    shallow_5000_1_dataset = Dataset.from_dict(shallow_5000_1)
    spinner.succeed("Training sets converted to datasets.")

    # Save the datasets
    spinner.start("Saving datasets to disk...")
    depth_50_50_dataset.save_to_disk("../data/trainingsets/depth_50_50")
    depth_50_100_dataset.save_to_disk("../data/trainingsets/depth_50_100")
    shallow_2500_1_dataset.save_to_disk("../data/trainingsets/shallow_2500_1")
    shallow_5000_1_dataset.save_to_disk("../data/trainingsets/shallow_5000_1")
    spinner.succeed("Datasets saved to disk.")
