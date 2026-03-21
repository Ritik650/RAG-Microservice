# Evaluating Retrieval and RAG Quality

Retrieval quality must be measured, not guessed. The foundation is a labeled
evaluation set: a list of questions, each annotated with a ground-truth answer and
the source documents that contain it. Thirty to fifty hand-labeled pairs are enough
to catch most regressions.

## Retrieval metrics

Recall@k asks: for what fraction of questions does a relevant chunk appear in the
top k results? Mean Reciprocal Rank (MRR) rewards putting the first relevant result
near the top: it averages 1 divided by the rank of the first relevant hit. nDCG,
normalized discounted cumulative gain, generalizes this by crediting every relevant
result with a logarithmic discount by position. Recall@k measures coverage, while
MRR and nDCG measure ranking quality.

## RAGAS metrics

RAGAS evaluates the generation side of a RAG pipeline. Faithfulness measures
whether the answer's claims are supported by the retrieved context. Answer relevancy
measures whether the answer actually addresses the question. Context precision asks
how much of the retrieved context was needed, and context recall asks how much of
the ground truth is covered by the retrieved context.

## Evaluation as a regression test

The most valuable practice is wiring evaluation into continuous integration. Each
change runs the retrieval eval; if recall@k or MRR drops below a fixed threshold,
the build fails. This turns retrieval quality into a regression test, exactly like
unit tests for code, and prevents silent quality erosion when swapping chunkers,
embedding models, or rerankers.
