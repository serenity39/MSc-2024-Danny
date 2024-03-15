"""Prepare the validation data for evaluating the BERT model."""

import ir_datasets
from halo import Halo
from preprocessing import map_ids_to_texts, training_set_to_dataset


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
    return training_set_to_dataset(validation_set, qid_to_text, docid_to_text)


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
