"""Generate subsets of the training data for model training."""

import random
from collections import defaultdict

import ir_datasets
from datasets import Dataset
from halo import Halo


def select_queries(qrels_by_query_id, num_queries, num_rels_per_query):
    """Select queries with the number of relevance judgments specified.

    Selects a specified number of queries with a given number of relevance
    judgments per query. Queries are selected randomly from the provided
    relevance judgments.

    Args:
        qrels_by_query_id (dict): dictionary containing relevance judgments by
            query id.
        num_queries (int): number of queries to select.
        num_rels_per_query (int): number of relevant judgments per query.

    Returns:
        list: list of selected queries with the specified relevance criteria.
    """
    selected_queries = []
    while len(selected_queries) < num_queries:
        query = random.choice(list(qrels_by_query_id.keys()))

        # Filter for relevance score of 1 or larger
        relevant_qrels = []
        for qrel in qrels_by_query_id[query]:
            if qrel[1] >= 1:
                relevant_qrels.append(qrel)

        # Only select queries with enough relevant documents
        if len(relevant_qrels) >= num_rels_per_query:
            selected_queries.append(query)
    return selected_queries


def map_ids_to_texts(dataset):
    """Map query and document IDs to their respective texts.

    This function creates a mapping from query and document IDs to their
    respective texts using the provided dataset from ir_datasets.

    Args:
        dataset: The dataset to extract the texts from.

    Returns:
        dict: A dictionary mapping query IDs to their texts.
        dict: A dictionary mapping document IDs to their texts.
    """
    qid_to_text = {}
    docid_to_text = {}

    for query in dataset.queries_iter():
        qid_to_text[query.query_id] = query.text

    for doc in dataset.docs_iter():
        docid_to_text[doc.doc_id] = doc.text

    return qid_to_text, docid_to_text


def create_training_set(dataset, num_queries, num_rels_per_query, seed=42):
    """Generates a training set with equal positive and negative examples.

    This function creates a training set by first randomly selecting a
    specified number of queries. For each selected query, it then samples a
    given number of relevant documents (positive examples) and an equal number
    of non-relevant documents (negative examples). Non-relevant documents are
    selected from the entire document set, excluding those marked as relevant
    for the query.

    Args:
        dataset: The dataset from which to generate the training set.
            This dataset should be loaded with `ir_datasets`.
        num_queries (int): The number of unique queries to include
            in the training set.
        num_rels_per_query (int): The number of relevant documents to sample
            for each selected query.
        seed (int, optional): A seed for the random number generator to ensure
            reproducibility. Defaults to 42.

    Returns:
        list of tuples: Each tuple in the list represents a query-document pair,
            consisting of the query ID, document ID, and a binary label
            indicating relevance (1 for relevant, 0 for non-relevant).
    """
    random.seed(seed)
    training_set = []
    qrels_by_query_id = defaultdict(list)
    docid_set = set()

    for qrel in dataset.qrels_iter():
        qrels_by_query_id[qrel.query_id].append((qrel.doc_id, qrel.relevance))
        docid_set.add(qrel.doc_id)

    # Select queries with the specified number of relevance judgments
    selected_queries = select_queries(
        qrels_by_query_id, num_queries, num_rels_per_query
    )

    for qid in selected_queries:
        positive_docs = random.sample(
            qrels_by_query_id[qid], num_rels_per_query
        )

        # Generate negative examples
        positive_doc_id = {doc[0] for doc in positive_docs}
        negative_doc_id = list(docid_set - positive_doc_id)
        negative_docs = random.sample(negative_doc_id, num_rels_per_query)

        # Add both positive and negative examples to the training set
        for pos_doc in positive_docs:
            doc_id = pos_doc[0]
            training_set.append((qid, doc_id, 1))

        for neg_docid in negative_docs:
            training_set.append((qid, neg_docid, 0))

    return training_set


def training_set_to_dataset(training_set, query_text_map, doc_text_map):
    """Convert the training set to a Huggingface dataset.

    Arg:
        training_set (list): list of tuples with training set information.
        query_text_map (dict): dictionary mapping query IDs to their texts.
        doc_text_map (dict): dictionary mapping document IDs to their texts.

    Returns:
        Dataset: Huggingface dataset with the training set information.
    """
    # Initialize lists to hold column data
    query_ids = []
    doc_ids = []
    query_texts = []
    doc_texts = []
    relevances = []

    # Populate the lists with data from the training set
    for query_id, doc_id, relevance in training_set:
        query_ids.append(query_id)
        doc_ids.append(doc_id)
        query_texts.append(query_text_map[query_id])
        doc_texts.append(doc_text_map[doc_id])
        relevances.append(relevance)

    # Create a dictionary that maps column names to data lists
    data_dict = {
        "query_id": query_ids,
        "doc_id": doc_ids,
        "query_text": query_texts,
        "doc_text": doc_texts,
        "relevance": relevances,
    }

    # Convert to Huggingface dataset
    dataset = Dataset.from_dict(data_dict)
    return dataset


def check_training_set(training_set):
    """Check the size and number of positive and negative examples.

    Args:
        training_set: The training set to check.
        Created with create_training_set.
    """
    # Total size of the training set
    total_size = len(training_set)

    # Count positive and negative examples
    num_positive = sum(1 for _, _, label in training_set if label == 1)
    num_negative = total_size - num_positive

    print(f"Total size of the training set: {total_size}")
    print(f"Number of positive examples: {num_positive}")
    print(f"Number of negative examples: {num_negative}")


if __name__ == "__main__":
    spinner = Halo(text="Loading datasets...", spinner="dots")

    # Load the datasets
    spinner.start()
    dataset_trec_2021 = ir_datasets.load("msmarco-passage-v2/trec-dl-2021")
    dataset_train = ir_datasets.load("msmarco-passage-v2/train")
    spinner.succeed("Datasets loaded.")

    # Map IDs to texts
    spinner.start("Mapping IDs to texts...")
    query_text_map_trec, doc_text_map_trec = map_ids_to_texts(dataset_trec_2021)
    query_text_map_train, doc_text_map_train = map_ids_to_texts(dataset_train)
    spinner.succeed("IDs mapped to texts.")

    # Generate training sets
    spinner.start("Creating depth-based training sets...")
    depth_based_50_50 = create_training_set(dataset_trec_2021, 50, 50)
    check_training_set(depth_based_50_50)
    depth_based_50_100 = create_training_set(dataset_trec_2021, 50, 100)
    check_training_set(depth_based_50_100)
    spinner.succeed("Depth-based training sets created.")

    spinner.start("Creating shallow-based training sets...")
    shallow_based_2500_1 = create_training_set(dataset_train, 2500, 1)
    check_training_set(shallow_based_2500_1)
    shallow_based_5000_1 = create_training_set(dataset_train, 5000, 1)
    check_training_set(shallow_based_5000_1)
    spinner.succeed("Shallow-based training sets created.")

    # Convert to Huggingface datasets
    spinner.start("Convert to Huggingface datasets...")
    hf_dataset_depth_50_50 = training_set_to_dataset(
        depth_based_50_50, query_text_map_trec, doc_text_map_trec
    )
    hf_dataset_depth_50_100 = training_set_to_dataset(
        depth_based_50_100, query_text_map_trec, doc_text_map_trec
    )
    hf_dataset_shallow_2500_1 = training_set_to_dataset(
        shallow_based_2500_1, query_text_map_train, doc_text_map_train
    )
    hf_dataset_shallow_5000_1 = training_set_to_dataset(
        shallow_based_5000_1, query_text_map_train, doc_text_map_train
    )
    spinner.succeed("Huggingface datasets created.")

    # Save the datasets
    spinner.start("Saving datasets...")
    hf_dataset_depth_50_50.save_to_disk("../data/hf_datasets/depth_based_50_50")
    hf_dataset_depth_50_100.save_to_disk(
        "../data/hf_datasets/depth_based_50_100"
    )
    hf_dataset_shallow_2500_1.save_to_disk(
        "../data/hf_datasets/shallow_based_2500_1"
    )
    hf_dataset_shallow_5000_1.save_to_disk(
        "../data/hf_datasets/shallow_based_5000_1"
    )
    spinner.succeed("Huggingface datasets saved.")
