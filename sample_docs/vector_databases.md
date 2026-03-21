# Vector Databases

A vector database stores high-dimensional embeddings and answers nearest-neighbor
queries over them. Unlike a relational database, its core operation is similarity
search: given a query vector, return the stored vectors closest to it under a
distance metric such as cosine similarity, dot product, or Euclidean distance.

## Approximate nearest neighbor indexes

Exact nearest-neighbor search scans every vector and becomes too slow at scale.
Vector databases therefore build approximate nearest neighbor (ANN) indexes. The
most widely used is HNSW — Hierarchical Navigable Small World — a layered proximity
graph. Upper layers act as coarse highways for long hops across the space, while
lower layers refine the search locally, giving logarithmic search complexity with
high recall.

## Payloads and filtering

Alongside each vector, databases store a payload of metadata such as the source
document, chunk index, and raw text. Queries can filter on payload fields, for
example restricting search to a single document or tenant. Qdrant stores payloads
as JSON and supports filtered search natively, which is essential for multi-tenant
retrieval systems.

## Quantization

To reduce memory, vectors can be quantized. Scalar quantization maps each float32
dimension to an int8, cutting memory roughly four times with minimal recall loss.
Binary quantization goes further, storing one bit per dimension, and works best for
high-dimensional embeddings. Quantized vectors accelerate search because more of
the index fits in RAM.
