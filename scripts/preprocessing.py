"""Preprocessing script making training sets for BERT."""

import random

import ir_datasets
import pandas as pd
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
                "query_id": qrel["query_id"],
                "doc_id": qrel["doc_id"],
                "relevance": 1 if qrel["relevance"] > 0 else 0,
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
    sampled_qrels = []
    all_doc_ids = set(docs.keys())

    for qrel in train_qrels:
        relevant_doc_id = qrel["doc_id"]
        non_relevant_docs = list(all_doc_ids - set([relevant_doc_id]))
        negative_samples = random.sample(non_relevant_docs, num_neg_samples)
        sampled_qrels.append(qrel)
        for doc_id in negative_samples:
            sampled_qrels.append(
                {"query_id": qrel["query_id"], "doc_id": doc_id, "relevance": 0}
            )
    return sampled_qrels


# Stratify the qrels to create the training sets
def create_stratified_sample(
    qrels, queries, docs, num_queries, num_rels_per_query, ratio
):
    """Create a stratified sample from the qrels.

    Args:
        qrels: The relevance judgments.
        queries: The queries.
        docs: The documents/passage.
        num_queries: Number of queries to sample.
        num_rels_per_query: Number of relevance judgments to sample per query.
        ratio: The ratio of relevant to non-relevant documents.

    Returns:
        list: A list of dictionaries containing the query text, document text,
              and relevance judgment.
    """
    # Create stratified subsets based on the ratio
    stratified_qrels = []
    selected_queries = random.sample(list(queries.keys()), num_queries)
    for query_id in selected_queries:
        query_qrels = [qrel for qrel in qrels if qrel["query_id"] == query_id]
        relevant_qrels = [
            qrel for qrel in query_qrels if qrel["relevance"] == 1
        ]
        non_relevant_qrels = [
            qrel for qrel in query_qrels if qrel["relevance"] == 0
        ]
        num_rel = int(num_rels_per_query * ratio)
        num_non_rel = num_rels_per_query - num_rel
        sampled_rel_qrels = random.sample(
            relevant_qrels, min(num_rel, len(relevant_qrels))
        )
        sampled_non_rel_qrels = random.sample(
            non_relevant_qrels, min(num_non_rel, len(non_relevant_qrels))
        )
        stratified_qrels.extend(sampled_rel_qrels + sampled_non_rel_qrels)
    # Convert IDs to text
    stratified_samples = []
    for qrel in stratified_qrels:
        stratified_samples.append(
            {
                "query_text": queries[qrel["query_id"]],
                "doc_text": docs[qrel["doc_id"]],
                "relevance": qrel["relevance"],
            }
        )
    return stratified_samples


# Function to load the text for queries and documents
def load_texts(dataset):
    """Load the text for queries and documents.

    Args:
        dataset_name: The name of the dataset to load (check ir_datasets).

    Returns:
        dict: A dictionary containing the queries and documents.
    """
    queries = {query.query_id: query.text for query in dataset.queries_iter()}
    docs = {doc.doc_id: doc.text for doc in dataset.docs_iter()}
    return queries, docs


# Function to create dataset with text data
def create_dataset_with_text(qrels, queries, docs):
    """Create a dataset with text data.

    Args:
        qrels: The relevance judgments.
        queries: The queries.
        docs: The documents/passage.

    Returns:
        list: A list of dictionaries containing the query text, document text,
              and relevance judgment.
    """
    data_with_text = []
    for qrel in qrels:
        query_text = queries.get(qrel["query_id"], None)
        doc_text = docs.get(qrel["doc_id"], None)
        if query_text is not None and doc_text is not None:
            data_with_text.append(
                {
                    "query_text": query_text,
                    "doc_text": doc_text,
                    "relevance": qrel["relevance"],
                }
            )
    return data_with_text


if __name__ == "__main__":
    spinner = Halo(text="Loading datasets...", spinner="dots")
    spinner.start()

    # Load datasets
    trec_dataset = ir_datasets.load("msmarco-passage-v2/trec-dl-2021")
    train_dataset = ir_datasets.load("msmarco-passage-v2/train")
    spinner.succeed("Datasets loaded!")

    # Load query and document texts
    spinner.start("Loading text data for trec-dl-2021 and train datasets...")
    trec_queries, trec_docs = load_texts(trec_dataset)
    train_queries, train_docs = load_texts(train_dataset)
    spinner.succeed("Text data loaded!")

    # Convert trec-dl-2021 relevance judgments to binary format
    spinner.start("Converting trec-dl-2021 datasets to binary format...")
    trec_qrels_binary = list(convert_to_binary(trec_dataset.qrels_iter()))
    spinner.succeed("trec-dl-2021 datasets converted to binary format.")

    # Negative sampling for train qrels
    spinner.start("Creating negative samples for train qrels...")
    combined_train_qrels = negative_sampling(
        train_dataset.qrels_iter(), train_docs
    )
    spinner.succeed("Negative samples created for train qrels.")

    # Creating depth-based training sets
    spinner.start("Creating depth-based 50/50 training set...")
    depth_50_50 = create_stratified_sample(
        trec_qrels_binary, trec_queries, trec_docs, 50, 50, 0.5
    )  # Adjust the ratio as needed
    spinner.succeed(
        f"Depth-based 50/50 training set created. Size: {len(depth_50_50)}"
    )

    spinner.start("Creating depth-based 50/100 training set...")
    depth_50_100 = create_stratified_sample(
        trec_qrels_binary, trec_queries, trec_docs, 50, 100, 0.5
    )  # Adjust the ratio as needed
    spinner.succeed(
        f"Depth-based 50/100 training set created. Size: {len(depth_50_100)}"
    )

    # Creating shallow-based training sets
    spinner.start("Creating shallow-based 2500/1 training set...")
    shallow_2500_1 = create_stratified_sample(
        combined_train_qrels, train_queries, train_docs, 2500, 1, 0.5
    )  # Adjust the ratio as needed
    spinner.succeed(
        f"Shallow-based 2500/1 training set created. "
        f"Size: {len(shallow_2500_1)}"
    )

    spinner.start("Creating shallow-based 5000/1 training set...")
    shallow_5000_1 = create_stratified_sample(
        combined_train_qrels, train_queries, train_docs, 5000, 1, 0.5
    )  # Adjust the ratio as needed
    spinner.succeed(
        f"Shallow-based 5000/1 training set created. "
        f"Size: {len(shallow_5000_1)}"
    )

    # Convert the training sets to Hugging Face datasets
    spinner.start("Converting training sets to Hugging Face datasets...")
    depth_50_50_dataset = Dataset.from_pandas(pd.DataFrame(depth_50_50))
    depth_50_100_dataset = Dataset.from_pandas(pd.DataFrame(depth_50_100))
    shallow_2500_1_dataset = Dataset.from_pandas(pd.DataFrame(shallow_2500_1))
    shallow_5000_1_dataset = Dataset.from_pandas(pd.DataFrame(shallow_5000_1))
    spinner.succeed("Training sets converted to Hugging Face datasets.")

    # Save the datasets
    spinner.start("Saving datasets to disk...")
    depth_50_50_dataset.save_to_disk("/path/to/save/depth_50_50")
    depth_50_100_dataset.save_to_disk("/path/to/save/depth_50_100")
    shallow_2500_1_dataset.save_to_disk("/path/to/save/shallow_2500_1")
    shallow_5000_1_dataset.save_to_disk("/path/to/save/shallow_5000_1")
    spinner.succeed("Datasets saved to disk.")
