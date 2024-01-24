"""Batch retrieval experiment using Pyserini with evaluation on MS MARCO."""

import csv

from pyserini.search.lucene import LuceneSearcher  # type: ignore


def perform_batch_retrieval(index_name, query_file_path, output_path):
    """Perform batch retrieval on a set of queries using Pyserini.

    Args:
        index_name (str): The name of the prebuilt index to use for searching.
        query_file_path (str): The path to the .tsv file containing queries.
        output_path (str): The path to the output file where
            results will be saved.

    Returns:
        None
    """
    # Initialize the Lucene searcher with the specified prebuilt index.
    searcher = LuceneSearcher.from_prebuilt_index(index_name)

    # Read queries from the .tsv file and write results to the output file.
    with open(query_file_path, "r") as query_file, open(
        output_path, "w"
    ) as output_file:
        tsv_reader = csv.reader(query_file, delimiter="\t")
        # Process each query and write results to the output
        # file in TREC format.
        for query_id, query in tsv_reader:
            print(f"Processing query {query_id}: {query}...")
            hits = searcher.search(query)
            for i, hit in enumerate(hits):
                output_file.write(
                    f"{query_id}\tQ0\t{hit.docid}\t{i+1}\t"
                    f"{hit.score:.5f}\tpassage2\n"
                )

    print(f"Batch retrieval complete. Results saved to {output_path}")


if __name__ == "__main__":
    index_name = "msmarco-v2-passage"
    query_file_path = "../data/queries/passv2_train_queries.tsv"
    output_path = "../data/runs/run.msmarco-v2-passage.txt"

    # Perform the batch retrieval
    perform_batch_retrieval(index_name, query_file_path, output_path)

    # Results:
    # map                     all     0.0635
    # P_10                    all     0.0146
    # recall_10               all     0.1440

    # Note: the run file did not contain all the queries as it got stuck and
    # therefore terminated before it could run to completion,
    # so the results might be different.
