"""Prepare the validation data for evaluating the BERT model."""

import ir_datasets
from datasets import Dataset
from halo import Halo
from preprocessing import map_ids_to_texts


def val_set_to_dataset(training_set):
    """Convert a set of data to a Huggingface dataset.

    Args:
        training_set (list): List of tuples with validation set information.

    Returns:
        Dataset: Huggingface dataset with the set information.
    """
    # Initialize lists to hold column data
    query_ids = []
    doc_ids = []
    query_texts = []
    doc_texts = []
    relevances = []

    # Populate the lists with data from the training or validation set
    for entry in training_set:
        query_id, doc_id, query_text, doc_text, relevance = entry
        query_ids.append(query_id)
        doc_ids.append(doc_id)
        query_texts.append(query_text)
        doc_texts.append(doc_text)
        relevances.append(relevance)

    # Create a dictionary that maps column names to data lists
    data_dict = {
        "query_id": query_ids,
        "doc_id": doc_ids,
        "query_text": query_texts,
        "doc_text": doc_texts,
        "labels": relevances,
    }

    # Convert to Huggingface dataset
    dataset = Dataset.from_dict(data_dict)

    return dataset


def create_validation_set(dataset, qid_to_text, docid_to_text):
    """Generates a validation set using development data.

    Args:
        dataset: The dataset to use.
        qid_to_text (dict): Dictionary mapping query IDs to their texts.
        docid_to_text (dict): Dictionary mapping document IDs to their texts.

    Returns:
        Dataset: Huggingface dataset with the validation set information.
    """
    validation_set = []
    for qrel in dataset.qrels_iter():
        relevance = qrel.relevance
        query_id = qrel.query_id
        doc_id = qrel.doc_id
        query_text = qid_to_text[query_id]
        doc_text = docid_to_text[doc_id]
        validation_set.append(
            (query_id, doc_id, query_text, doc_text, relevance)
        )

    # Convert the list of tuples to a Huggingface dataset
    validation_set = val_set_to_dataset(validation_set)
    return validation_set


if __name__ == "__main__":
    spinner = Halo(text="Loading dataset from ir_datasets...", spinner="dots")

    spinner.start()
    # Load the dataset
    dataset = ir_datasets.load("msmarco-passage-v2/dev2")
    spinner.succeed("Dataset loaded")

    # Create a dictionary mapping query IDs to their texts
    spinner.start("Mapping query IDs to their texts...")
    query_text_map, doc_text_map = map_ids_to_texts(dataset)
    spinner.succeed("Query IDs mapped to their texts")

    # Create the validation set
    spinner.start("Creating the validation set...")
    validation_set = create_validation_set(
        dataset, query_text_map, doc_text_map
    )
    spinner.succeed("Validation set created")

    # Save the validation set to disk
    spinner.start("Saving the validation set to disk...")
    validation_set.save_to_disk("../data/hf_datasets/validation_set")
    spinner.succeed("Validation set saved to disk")
