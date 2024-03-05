"""Prepare the validation data for evaluating the BERT model."""

import csv
import os

import ir_datasets
from halo import Halo


def create_validation_set(dataset, output_csv_path):
    """Create a validation dataset for evaluating model.

    Format the validation dataset as a CSV file with the following columns:
    - Query text
    - Document text
    - Relevance score


    Args:
        dataset (Dataset): Dataset object loaded from ir_datasets.
        output_csv_path (str): Path to save the validation CSV.

    Returns:
        file: A file containing the validation dataset in CSV format.
    """
    # Create dictionaries to map query and document IDs to text
    qid_to_text = {}
    docid_to_text = {}

    spinner = Halo(
        text="Populating dictionaries with queries and doc text...",
        spinner="dots",
    )
    spinner.start()
    # Populate the dictionary with the queries and documents
    for query in dataset.queries_iter():
        qid_to_text[query.query_id] = query.text
    for doc in dataset.docs_iter():
        docid_to_text[doc.doc_id] = doc.text
    spinner.succeed("Dictionaries populated!")

    spinner.start("Prepare the validation set tuples...")
    # Prepare the validation set tuples
    validation_set = []
    for qrel in dataset.qrels_iter():
        validation_set.append((qrel.query_id, qrel.doc_id, qrel.relevance))
    spinner.succeed("Validation set tuples prepared!")

    spinner.start("Write validation set to a CSV file...")
    # Write validation set to a CSV file
    with open(output_csv_path, "w", newline="", encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        for query_id, doc_id, relevance in validation_set:
            query_text = qid_to_text.get(query_id, "")
            doc_text = docid_to_text.get(doc_id, "")
            csv_writer.writerow([query_text, doc_text, relevance])
    spinner.succeed("Validation set written to CSV file!")


if __name__ == "__main__":
    spinner = Halo(text="Load dataset...", spinner="dots")
    spinner.start()
    # Load the MS MARCO dataset for validation
    dataset_validation = ir_datasets.load(
        "msmarco-passage-v2/trec-dl-2021/judged"
    )
    spinner.succeed("Dataset loaded!")

    # Define path to save the validation data
    validation_csv_path = "../data/validationdata/validation_data.csv"
    os.makedirs(validation_csv_path, exist_ok=True)

    # Create the validation set and save to CSV
    create_validation_set(dataset_validation, validation_csv_path)
