"""Evaluate model using Pyserini and TREC Eval."""

import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import ir_datasets  # noqa: E402
import torch  # noqa: E402
from pyserini.search.lucene import LuceneSearcher  # noqa: E402
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import logging as tf_logging  # noqa: E402

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Comment out if you want to see the warnings
tf_logging.set_verbosity_error()


# Ensure the GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the prebuilt index
searcher = LuceneSearcher.from_prebuilt_index("msmarco-v2-passage")

# Load model and tokenizer
model_path = "../data/results/models/early_stopping/depth_based_50_200"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# Load queries
query_dataset = ir_datasets.load("msmarco-passage-v2/dev1")
queries = {}
for query in query_dataset.queries_iter():
    queries[query.query_id] = query.text
logging.info(f"Loaded {len(queries)} queries.")

# Setting up the run file
run_file_path = "../data/results/runs/early_stopping/depth_based_50_200_run.txt"
run_name = "es_depth_based_50_200"

logging.info("Evaluating the model...")
with open(run_file_path, "w") as run_file:
    for query_id, query in queries.items():
        # Search for the query
        hits = searcher.search(query)

        # Rerank the hits
        reranked_docs = []
        for hit in hits[:100]:
            doc = searcher.doc(hit.docid)
            doc_text = doc.raw()
            inputs = tokenizer.encode_plus(
                query,
                doc_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )

            # Move each tensor in the inputs dictionary to the specified device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            score = probabilities[:, 1].item()
            reranked_docs.append((hit.docid, score))

        # Sort the documents by score for this query
        reranked_docs.sort(key=lambda x: x[1], reverse=True)

        # Write the reranked results to the run file
        for i, (docid, score) in enumerate(reranked_docs):
            run_file.write(f"{query_id} Q0 {docid} {i+1} {score} {run_name}\n")
logging.info("Model evaluation complete.")

logging.info(f"Run file saved to: {run_file_path}")
