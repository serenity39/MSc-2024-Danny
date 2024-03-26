"""Test model"""

import json
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

text_pairs2 = [
    (
        "What is baron nashor?",
        "Debuff Immunity: Baron Nashor is immune to all crowd control, except stasis. Additionally, Baron Nashor's stats cannot be modified by any means. Baron's Gaze: Baron Nashor takes 50% reduced damage from the unit that it has most recently attacked for 8 seconds, reduced to 4. 5 seconds after Baron Nashor is slain.",
    ),
    (
        "Do monkeys eat bananas?",
        "Monkey diet. In fact, wild monkeys do not eat bananas! The banana does not grow naturally; They are planted by humans, so wild monkeys don’t even get a chance to eat them. In the wild, most monkeys are omnivorous, which means they eat both trees and meat. And all monkeys eat about the same thing. Depending on their habitat, things may change, but all monkeys eat fruits, leaves, seeds, nuts, flowers, vegetables, and insects.",
    ),
    ("Do monkeys eat bananas?", "Monkeys are yellow."),
    (
        "what is the capital of France",
        "No. Madrid is the capital of Spain. Baghdad is the capital of Iraq. What is the capital of Spain and France? There is no such thing as 'the capital of Spain and France'; they are two separate countries. The capital of Spain is Madrid, and the capital of France is Paris.",
    ),
    ("what is the capital of France", "Madrid is the capital of Spain."),
    (
        "How fast can a cheetah run?",
        "How fast can a cheetah run. King of speed, cheetah is undoubtedly the fastest land animal on earth but still how fast can a cheetah run ? This born to run machine can easily reach speeds off about 70 mph (113 kph) and all it takes for a cheetah to reach its potential is a few strides.",
    ),
    ("How fast can a cheetah run?", "Wild african cats are slow."),
    ("How fast can a cheetah run?", "Wild african cats are really fast."),
]

print("Model predictions:")
for query, segment in text_pairs2:
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
            f"Similarity between: '{query}' and '{segment}' is:\n {similarity_score}"
        )

print("\nBM25 hits:")

searcher = LuceneSearcher.from_prebuilt_index("msmarco-v2-passage")

for query, _ in text_pairs2:
    hits = searcher.search(query)
    for hit in hits[:1]:
        doc = searcher.doc(hit.docid)
        doc_text = doc.raw()
        passage = json.loads(doc_text)["passage"]
        print(f"Query: '{query}' \n Passage: '{passage}'")
