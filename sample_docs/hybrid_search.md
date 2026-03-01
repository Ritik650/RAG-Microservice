# Hybrid Search and Reranking

Dense vector search retrieves passages by semantic similarity, but it can miss exact
keyword matches such as product names, error codes, or rare terms. Sparse methods
like BM25 excel at exact lexical matching but ignore meaning. Hybrid search runs both
and fuses their results, capturing the strengths of each.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) is a simple, robust way to combine ranked lists. Each
document receives a score based on its rank in each list, and the scores are summed.
Because it uses ranks rather than raw scores, RRF does not require the dense and
sparse scores to be on the same scale, which makes it easy to deploy.

## Cross-encoder reranking

After hybrid retrieval produces a candidate set, a cross-encoder reranker can re-score
the top-k passages by jointly encoding the query and each passage. This is more
expensive than the initial retrieval but far more precise, and it is applied only to a
small candidate set. Measuring retrieval quality before and after reranking typically
shows a clear precision improvement, which makes a compelling before/after chart.

## Measuring retrieval quality

Retrieval quality is measured against a labeled question-to-answer set using metrics
such as recall@k, mean reciprocal rank, and nDCG. Answer quality is measured with
metrics like faithfulness and context precision. Wiring these metrics into continuous
integration turns retrieval quality into a regression test.
