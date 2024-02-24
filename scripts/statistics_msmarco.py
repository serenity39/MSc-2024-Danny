"""Statistics on relevance judgements."""


def load_qrels(filepath):
    """Loads relevance judgements (qrels) from a TSV file.

    Each line in the file contains the following fields: query ID, iteration,
    document ID, and relevance score, separated by tabs.

    Args:
        filepath (str): The path to the TSV file containing
                        relevance judgements.

    Returns:
        list of tuples: A list where each tuple represents a judgement
        (query_id, doc_id, relevance).
    """
    qrels = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            query_id, _, doc_id, relevance = line.strip().split("\t")
            qrels.append((query_id, doc_id, int(relevance)))
    return qrels


def analyze_qrels(qrels):
    """Analyzes the loaded qrels to gather statistics.

    Computes the total number of relevance judgements,
    the distribution of judgements per query,
    and the number of unique documents judged
    and their respective judgement counts.

    Args:
        qrels (list of tuples): A list of qrels loaded from the TSV file.

    Returns:
        dict: A dictionary with statistics including total judgements,
              judgements per query, and unique documents count.
    """
    total_judgements = len(qrels)
    judgements_per_query = {}
    unique_documents = set()

    for query_id, doc_id, _ in qrels:
        judgements_per_query[query_id] = (
            judgements_per_query.get(query_id, 0) + 1
        )
        unique_documents.add(doc_id)

    stats = {
        "total_judgements": total_judgements,
        "judgements_per_query": judgements_per_query,
        "unique_documents_count": len(unique_documents),
    }

    return stats


def write_statistics_to_file(stats, output_filepath):
    """Writes the gathered statistics to a specified file.

    Args:
        stats (dict): A dictionary containing the statistics.
        output_filepath (str): The file path to write the statistics to.
    """
    with open(output_filepath, "w", encoding="utf-8") as file:
        file.write(f"Total number of judgements: {stats['total_judgements']}\n")
        file.write(
            f"Number of unique documents judged: "
            f"{stats['unique_documents_count']}\n"
        )
        file.write("Judgements per query:\n")
        for query, count in stats["judgements_per_query"].items():
            file.write(f"Query ID {query}: {count}\n")


if __name__ == "__main__":
    dev1_filepath = "../data/passv2_dev_qrels.tsv"
    dev2_filepath = "../data/passv2_dev2_qrels.tsv"
    output_filepath = "statistics"

    # Load and analyze Dev 1 qrels
    dev1_qrels = load_qrels(dev1_filepath)
    dev1_stats = analyze_qrels(dev1_qrels)
    write_statistics_to_file(dev1_stats, output_filepath + "_dev1.txt")

    # Load and analyze Dev 2 qrels
    dev2_qrels = load_qrels(dev2_filepath)
    dev2_stats = analyze_qrels(dev2_qrels)
    write_statistics_to_file(dev2_stats, output_filepath + "_dev2.txt")

    print("Statistics for Dev 1 and Dev 2 sets have been written to:")
    print(f"'{output_filepath}_dev1.txt' and")
    print(f"'{output_filepath}_dev2.txt' respectively.")
