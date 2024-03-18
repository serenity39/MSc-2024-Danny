"""Evaluate model using Pyserini and TREC Eval."""

import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import ir_datasets  # noqa: E402
import torch  # noqa: E402
from pyserini.search.lucene import LuceneSearcher  # noqa: E402
from transformers import BertForSequenceClassification, BertTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Ensure the GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the prebuilt index
searcher = LuceneSearcher.from_prebuilt_index("msmarco-v2-passage")

# Load model and tokenizer
model_path = "../data/results/models/depth_based_50_100/"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# Load queries
query_dataset = ir_datasets.load("msmarco-passage-v2/dev1")
queries = {}
for query in query_dataset.queries_iter():
    queries[query.query_id] = query.text

# Setting up the run file
run_file_path = "../data/results/runs/depth_50_100_run.txt"
run_name = "depth_50_100"

with open(run_file_path, "w") as run_file:
    for query_id, query in queries.items():
        # Search for the query
        hits = searcher.search(query)

        # Rerank the hits
        reranked_docs = []
        for hit in hits[:1000]:
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
            score = outputs.logits[:, 1]
            reranked_docs.append((hit.docid, score.item()))

        # Sort the documents by score for this query
        reranked_docs.sort(key=lambda x: x[1], reverse=True)

        # Write the reranked results to the run file
        for i, (docid, score) in enumerate(reranked_docs):
            run_file.write(f"{query_id} Q0 {docid} {i+1} {score} {run_name}\n")

print(f"Run file saved to: {run_file_path}")
