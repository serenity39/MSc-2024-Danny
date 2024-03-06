"""Generate subsets of the training data for model training."""

import csv
import random
from collections import defaultdict

import ir_datasets
from halo import Halo


def create_training_set(
    dataset, num_queries, num_rels_per_query, seed=42, create_non_relevant=False
):
    """Create tuples with information for creating training sets.

    Creates a tuple with information to make the training set by randomly
    selecting a specified number of queries and relevance judgments from
    a given dataset.

    Args:
        dataset_name (str): Name of the dataset to load.
        num_queries (int): Number of queries to randomly select.
        num_rels_per_query (int): Number of relevance judgments per
                                selected query.
        seed (int): Random seed for reproducibility.

    Returns:
        list of tuples: A list of (query_id, doc_id, relevance) tuples.
    """
    # Seed the random number generator for reproducibility
    random.seed(seed)
    training_set = []

    # Create a dictionary to hold relevance judgments by query id
    qrels_by_query_id = defaultdict(list)

    # Populate the dictionary with the relevance judgments
    for qrel in dataset.qrels_iter():
        qrels_by_query_id[qrel.query_id].append(qrel)

    # Randomly select queries
    selected_queries = []
    while len(selected_queries) < num_queries:
        query = random.choice(list(qrels_by_query_id.keys()))
        # Only select queries with enough relevance judgments
        if (
            len(qrels_by_query_id[query]) >= num_rels_per_query
            and qrels_by_query_id[query].relevalance <= 1
        ):
            selected_queries.append(query)

    # Create the training set based on the selected queries
    for query_id in selected_queries:
        selected_qrels = random.sample(
            qrels_by_query_id[query_id], num_rels_per_query
        )
        for qrel in selected_qrels:
            training_set.append((qrel.query_id, qrel.doc_id, qrel.relevance))

    # Create non-relevant examples by swapping passages between queries
    if create_non_relevant:
        for i in range(len(training_set)):
            # Randomly select a different query
            different_query_id = random.choice(selected_queries)
            while different_query_id == training_set[i][0]:
                different_query_id = random.choice(selected_queries)

            # Get the list of documents that are relevant to the different query
            relevant_docs = []
            for qrel in qrels_by_query_id[different_query_id]:
                relevant_docs.append(qrel.doc_id)

            # Randomly select a relevant document from the different query
            swapped_doc_id = random.choice(relevant_docs)

            # Replace the document in the training set with the swapped document
            training_set[i] = (training_set[i][0], swapped_doc_id, 0)

    return training_set


if __name__ == "__main__":
    spinner = Halo(text="Creating training sets...", spinner="dots")
    spinner.start()

    # Load the MS MARCO dataset
    dataset_trec_2021 = ir_datasets.load("msmarco-passage-v2/trec-dl-2021")
    dataset_train = ir_datasets.load("msmarco-passage-v2/train")

    # Create two training sets tuples for depth-based and shallow-based
    depth_based_50_50 = create_training_set(dataset_trec_2021, 50, 50)
    depth_based_50_100 = create_training_set(dataset_trec_2021, 50, 100)
    shallow_based_2500_1 = create_training_set(
        dataset_train, 2500, 1, create_non_relevant=True
    )
    shallow_based_5000_1 = create_training_set(
        dataset_train, 5000, 1, create_non_relevant=True
    )
    spinner.succeed("Training sets created!")

    # Print out the lengths of the datasets as a simple check
    print(f"Depth-based 50/50: {len(depth_based_50_50)}")
    print(f"Depth-based 50/100: {len(depth_based_50_100)}")
    print(f"Shallow-based 2500/1: {len(shallow_based_2500_1)}")
    print(f"Shallow-based 5000/1: {len(shallow_based_5000_1)}")

    # Depth-based datasets
    spinner = Halo(text="Creating Depth-Based datasets...", spinner="dots")

    spinner.start("Creating Depth-Based 50/50 dataset...")
    # Create dictionaries to map query and document IDs to text
    qid_to_text_depth = {}
    docid_to_text_depth = {}

    for query in dataset_trec_2021.queries_iter():
        qid_to_text_depth[query.query_id] = query.text

    for doc in dataset_trec_2021.docs_iter():
        docid_to_text_depth[doc.doc_id] = doc.text

    # Prepare the depth-based datasets for training
    with open(
        "../data/trainingsets/inputs_depth_50_50.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as inputs_file:
        csv_writer = csv.writer(inputs_file)
        for query_id, doc_id, relevance in depth_based_50_50:
            query_text = qid_to_text_depth[query_id]
            doc_text = docid_to_text_depth[doc_id]
            csv_writer.writerow([query_text, doc_text, relevance])
    spinner.succeed("Depth-Based 50/50 dataset created!")

    spinner.start("Creating Depth-Based 50/100 dataset...")
    with open(
        "../data/trainingsets/inputs_depth_50_100.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as inputs_file:
        csv_writer = csv.writer(inputs_file)
        for query_id, doc_id, relevance in depth_based_50_100:
            query_text = qid_to_text_depth[query_id]
            doc_text = docid_to_text_depth[doc_id]
            csv_writer.writerow([query_text, doc_text, relevance])
    spinner.succeed("Depth-Based 50/100 dataset created!")

    # Shallow-based datasets
    spinner = Halo(text="Creating Shallow-Based datasets...", spinner="dots")

    spinner.start("Creating Shallow-Based 2500/1 dataset...")
    # Create dictionaries to map query and document IDs to text
    qid_to_text_shallow = {}
    docid_to_text_shallow = {}

    for query in dataset_train.queries_iter():
        qid_to_text_shallow[query.query_id] = query.text

    for doc in dataset_train.docs_iter():
        docid_to_text_shallow[doc.doc_id] = doc.text

    with open(
        "../data/trainingsets/inputs_shallow_2500_1.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as inputs_file:
        csv_writer = csv.writer(inputs_file)
        for query_id, doc_id, relevance in shallow_based_2500_1:
            query_text = qid_to_text_shallow[query_id]
            doc_text = docid_to_text_shallow[doc_id]
            csv_writer.writerow([query_text, doc_text, relevance])
    spinner.succeed("Shallow-Based 2500/1 dataset created!")

    spinner.start("Creating Shallow-Based 5000/1 dataset...")
    with open(
        "../data/trainingsets/inputs_shallow_5000_1.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as inputs_file:
        csv_writer = csv.writer(inputs_file)
        for query_id, doc_id, relevance in shallow_based_5000_1:
            query_text = qid_to_text_shallow[query_id]
            doc_text = docid_to_text_shallow[doc_id]
            csv_writer.writerow([query_text, doc_text, relevance])
    spinner.succeed("Shallow-Based 5000/1 dataset created!")
