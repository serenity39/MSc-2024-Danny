"""BERT reranking model."""

from pygaggle.rerank.base import Query, hits_to_texts
from pygaggle.rerank.transformer import MonoBERT
from pyserini.search import LuceneSearcher

reranker = MonoBERT()

query = Query("who proposed the geocentric theory")

searcher = LuceneSearcher.from_prebuilt_index("msmarco-passage")
hits = searcher.search(query.text)

texts = hits_to_texts(hits)

# Passages prior to re-ranking
print("Passages prior to re-ranking:")
for i in range(0, 10):
    print(
        f'{i+1:2} {texts[i].metadata["docid"]:15} {texts[i].score:.5f} '
        f"{texts[i].text}"
    )

# Re-rank
reranked = reranker.rerank(query, texts)

print("\nPassages after re-ranking:")
for i in range(0, 10):
    print(
        f'{i+1:2} {reranked[i].metadata["docid"]:15} {reranked[i].score:.5f} '
        f"{reranked[i].text}"
    )
