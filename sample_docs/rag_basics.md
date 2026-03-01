# Retrieval-Augmented Generation (RAG) Basics

Retrieval-Augmented Generation combines a retriever with a generative language
model. Instead of relying only on what the model memorized during training, a RAG
system first retrieves relevant passages from an external corpus and then conditions
the model's answer on those passages. This grounds answers in source documents and
makes it possible to cite where each claim came from.

## The ingestion pipeline

Before anything can be retrieved, documents must be ingested. A typical pipeline
splits each document into overlapping chunks, embeds every chunk into a dense
vector, and upserts those vectors into a vector database. Overlap between adjacent
chunks preserves context that would otherwise be lost at a hard cut. Ingestion
should be idempotent: re-running it on the same document must not create duplicate
chunks, which is usually achieved by deriving a deterministic id from the document
source and chunk index.

## Chunking

Chunk size is a trade-off. Chunks that are too large dilute the embedding and hurt
retrieval precision; chunks that are too small lose the surrounding context needed
to answer a question. A common starting point is a few hundred characters or tokens
per chunk with a modest overlap of 10-20 percent.

## Why cite sources

Every answer a RAG system produces should reference the chunks it used. Citations
let a reader verify claims, and they turn an opaque answer into an auditable one.
This is the difference between a demo and a trustworthy document question-answering
feature.
