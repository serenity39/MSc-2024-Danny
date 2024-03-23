"""Test model"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch  # noqa: E402
from pyserini.search.lucene import LuceneSearcher
from transformers import BertForSequenceClassification, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your fine-tuned model and tokenizer
model_path = "../data/results/models/depth_based_50_100/"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model = model.to(device)
model.eval()

# Example text pairs
text_pairs = [
    (
        "What is baron nashor?",
        "Baron Nashor is the most powerful neutral monster of the summoners rift.",
    ),
    (
        "What is baron nashor?",
        "Baron Nashor is a powerful monster that requires multiple champions to defeat.",
    ),
    ("What is baron nashor?", "Monster in League of Legends."),
    ("What is baron nashor?", "Monkeys likes bananas. They are yellow."),
    (
        "Do monkeys eat bananas?",
        "Chimpanzees are omnivores. They like bananas and climb trees.",
    ),
    ("Do monkeys eat bananas?", "Monkeys are yellow."),
    (
        "What is the capital of France?",
        "Paris is the capital of the european country France.",
    ),
    ("What is the capital of France?", "Paris and venezuela are in Europe."),
    ("How fast can a cheetah run?", "Cheetahs are the fastest land animal."),
    ("How fast can a cheetah run?", "Cheetahs are slow."),
    (
        "How fast can a cheetah run?",
        "Cheetahs can run up to 60 miles per hour.",
    ),
    ("How fast can a cheetah run?", "The fastet cat can run 60 mph."),
]

print("Model predictions:")
for query, segment in text_pairs:
    # Tokenize inputs
    inputs = tokenizer.encode_plus(
        query,
        segment,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )

    # Move each tensor in the inputs dictionary to the specified device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Make predictions without calculating gradients
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        similarity_score = probabilities[:, 1].item()
        print(
            f"Similarity between: '{query}' and '{segment}' is {similarity_score}"
        )

print("BM25 predictions:")

searcher = LuceneSearcher.from_prebuilt_index("msmarco-v2-passage")

for query, segment in text_pairs:
    hits = searcher.search(query)
    for hit in hits[:1]:
        doc = searcher.doc(hit.docid)
        doc_text = doc.raw()
        segment = doc_text[1]
        print(f"Similarity between: '{query}' and '{segment}' is {hit.score}")
