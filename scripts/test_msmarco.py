import random

import ir_datasets
from datasets import Dataset
from halo import Halo

DATASET_PATH = "../data/hf_datasets/msmarco_train/hf_training"

# Set seed
random.seed(42)

# Load the MSMARCO dataset
dataset = ir_datasets.load("msmarco-passage-v2/train")

# Get the queries
spinner = Halo(text="Loading queries...", spinner="dots")
spinner.start()
queries = {query.query_id: query.text for query in dataset.queries_iter()}
spinner.succeed(f"Queries loaded: {len(queries)}")

# Get the documents
spinner.start("Loading documents...")
docs = {doc.doc_id: doc.text for doc in dataset.docs_iter()}
spinner.succeed(f"Documents loaded: {len(docs)}")

# Get qrels (relevance judgments) and store them in a list
spinner.start("Loading relevance judgments...")
qrels = list(dataset.qrels_iter())
spinner.succeed(f"Relevance judgments loaded: {len(qrels)}")

# Prepare a set of all doc IDs for sampling of negative examples
all_doc_ids = set(docs.keys())

# Prepare a set of relevant doc IDs for sampling of negative examples
relevant_doc_ids = {qrel.doc_id for qrel in qrels}
negative_doc_ids = list(all_doc_ids - relevant_doc_ids)

# Shuffle the qrels to ensure random selection
random.shuffle(qrels)

# Limit the qrels to the desired number
desired_qrels_count = 100000
qrels = qrels[:desired_qrels_count]

# Prepare data for the Hugging Face dataset
spinner.start("Preapring data...")
data = []
for qrel in qrels:
    # Get the text of the query and document using their respective IDs
    query_text = queries.get(qrel.query_id)
    positive_doc_text = docs.get(qrel.doc_id)

    # Only add the example if both the query and document texts are found
    if query_text and positive_doc_text:
        # Add the positive example
        data.append(
            {
                "query_id": qrel.query_id,
                "doc_id": qrel.doc_id,
                "query_text": query_text,
                "doc_text": positive_doc_text,
                "label": qrel.relevance,
            }
        )

        # Sample a negative example
        negative_doc_id = random.choice(negative_doc_ids)
        negative_doc_text = docs[negative_doc_id]
        negative_doc_ids.remove(negative_doc_id)

        # Add the negative example
        data.append(
            {
                "query_id": qrel.query_id,
                "doc_id": negative_doc_id,
                "query_text": query_text,
                "doc_text": negative_doc_text,
                "label": 0,
            }
        )
spinner.succeed(f"Data prepared: {len(data)} examples")

# Shuffle the data to mix positive and negative examples
random.shuffle(data)

# Convert the data to a Hugging Face dataset
spinner.start("Converting to Hugging Face dataset...")
hf_dataset = Dataset.from_dict(
    {
        "query_id": [example["query_id"] for example in data],
        "doc_id": [example["doc_id"] for example in data],
        "query_text": [example["query_text"] for example in data],
        "doc_text": [example["doc_text"] for example in data],
        "label": [example["label"] for example in data],
    }
)
spinner.succeed("Data conversion complete.")

spinner.start("Saving dataset...")
hf_dataset.save_to_disk(DATASET_PATH)
spinner.succeed(f"Dataset saved to disk at: {DATASET_PATH}")

# Print the first few examples to verify
print(hf_dataset[:5])
