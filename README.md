# RAG Learning Project

A hands-on implementation and study of a Retrieval-Augmented Generation
(RAG) system using Python, ChromaDB, Sentence Transformers, and LLM
APIs. The project is being developed incrementally: each retrieval
technique is implemented, evaluated against a manually labelled test
set, and compared with the existing pipeline.

> **Project status:** Core RAG pipeline implemented. Retrieval
> optimisation experiments completed to date include query rewriting,
> multi-query retrieval with Reciprocal Rank Fusion (RRF), contextual
> retrieval, and Cross-Encoder reranking. Hybrid search and later
> production-oriented stages remain planned work.

------------------------------------------------------------------------

## Table of contents

-   [Objectives](#objectives)
-   [Technology stack](#technology-stack)
-   [Current architecture](#current-architecture)
-   [Implemented components](#implemented-components)
-   [Retrieval evaluation](#retrieval-evaluation)
-   [Experiments and findings](#experiments-and-findings)
-   [Current best-performing
    approach](#current-best-performing-approach)
-   [Limitations and open questions](#limitations-and-open-questions)
-   [Roadmap](#roadmap)
-   [Learning approach](#learning-approach)

------------------------------------------------------------------------

## Objectives

The project has two connected objectives:

1.  Build a working RAG pipeline that ingests documents, retrieves
    relevant passages, and uses an LLM to generate grounded responses.
2.  Understand the design choices behind RAG systems by implementing
    retrieval techniques individually and measuring their effects rather
    than adopting them without evaluation.

The initial knowledge base contains two technical documents:

-   **Self-Healing WAF** --- a hybrid web application firewall project.
-   **TurboQuant** --- a local chatbot / efficient LLM inference
    project.

The corpus is deliberately small and controlled so that retrieval
behaviour can be inspected at chunk level.

## Technology stack

  Component                     Technology / choice
  ----------------------------- --------------------------
  Language                      Python
  Vector database               ChromaDB
  Embedding model               `BAAI/bge-small-en-v1.5`
  Embedding dimension           384
  Vector distance metric        L2
  Local / API model interface   OpenAI-compatible client
  Hosted inference              Groq API
  Local inference               Ollama
  Local models tested           Qwen 2.5 7B and 14B
  Planned API layer             FastAPI

The same embedding model is used for document chunks and incoming
queries. The ChromaDB collection uses L2 distance; lower vector
distances indicate closer vectors.

The project has used Groq for hosted LLM calls and has also been
configured to call an Ollama server hosted on a Windows PC from a Mac
over a private local network. The local setup is intended to allow the
Python project to run on the Mac while inference is served by the
Windows machine.

------------------------------------------------------------------------

## Current architecture

At a high level, the pipeline follows these stages:

1.  Load source documents.
2.  Split each document into sentence-aware chunks.
3.  Optionally contextualise each chunk using document-level analysis.
4.  Generate embeddings for the chunks.
5.  Store chunk IDs, text, embeddings, and metadata in ChromaDB.
6.  Embed the user's query.
7.  Retrieve candidate chunks from ChromaDB.
8.  Optionally apply retrieval optimisation, such as query rewriting or
    Cross-Encoder reranking.
9.  Filter retrieved results using a distance threshold.
10. Construct context and a prompt.
11. Send the prompt to the generation model.
12. Return a grounded answer, or abstain when the retrieval stage does
    not find suitable context.

The core ingestion and retrieval responsibilities have been separated
into modules such as `ingest.py` and `search_retrieve.py`, with prompt
templates stored separately.

------------------------------------------------------------------------

## Implemented components

### 1. Document loading and chunking

A sentence-based chunker was implemented using sentence-boundary
splitting:

-   Sentences are grouped until the configured chunk-size limit is
    reached.
-   Long sentences are handled separately.
-   Chunk order is preserved.
-   Chunks are represented as dictionaries containing identifiers and
    text.
-   Overlap can be configured.

The implementation uses sentence boundaries rather than blindly cutting
text at fixed character positions. This helps preserve the meaning of
passages.

For contextual-retrieval experiments, overlap was set to **0** to avoid
repeated content affecting the comparison. The same chunk size and
boundaries should be used when comparing baseline and contextualised
corpora.

### 2. Embedding generation

Embedding generation is a separate step from chunking.

The `embed_chunks` stage encodes chunk text with
`BAAI/bge-small-en-v1.5` and attaches the resulting vector to the
corresponding chunk record. Keeping chunking and embedding separate
makes each stage easier to inspect, test, and optimise.

For contextual retrieval, the contextualised text is embedded rather
than the unchanged original text.

### 3. ChromaDB storage and retrieval

Chunks are stored with:

-   Chunk ID
-   Document text
-   Embedding
-   Source metadata

The retrieval stage embeds the query using the same embedding model and
requests the nearest documents from ChromaDB.

The project has also used a distance threshold to decide whether
retrieved material is sufficiently close to the query. The threshold
used in the initial RAG pipeline was 0.8; this is an experimental
setting, not a universal threshold.

A key implementation consideration is to keep baseline and
contextualised data in separate collections when comparing them. Their
chunk IDs and chunk boundaries should remain aligned.

### 4. Context construction and generation

The retrieved chunks are assembled into context and passed to a prompt
that instructs the generation model to answer using the supplied
material.

The pipeline has been connected to Groq chat completions. Local Ollama
inference has also been explored through an OpenAI-compatible client
interface.

The intended behaviour is to avoid answering from unsupported context.
The current retrieval evaluation does not yet fully measure abstention
quality; see [Limitations and open
questions](#limitations-and-open-questions).

### 5. Query rewriting

A query-rewriting stage was implemented using an LLM.

Its instructions are to:

-   Preserve the original meaning and intent.
-   Use relevant technical terminology where appropriate.
-   Avoid adding assumptions or unsupported information.
-   Return only the rewritten query.

The rewritten query is used for retrieval. The original query remains
the user's question and should still be used when constructing the final
answer prompt where appropriate.

### 6. Multi-query retrieval and Reciprocal Rank Fusion

A multi-query retrieval experiment was implemented:

1.  Generate several alternative retrieval queries from the original
    question.
2.  Run vector retrieval for each alternative.
3.  Record the rank of each chunk in each result list.
4.  Combine rankings using Reciprocal Rank Fusion (RRF).

The RRF score is:

$$
RRF(d) = \sum_{r \in R} \frac{1}{k + \operatorname{rank}_r(d)}
$$

Where:

-   \(d\) is a document / chunk.
-   \(R\) is the set of ranked result lists.
-   (rank_r(d)) is the rank of document (d) in list (r).
-   \(k\) is a smoothing constant; the experiment used 60.

RRF uses rank positions rather than raw vector distances. A chunk
appearing near the top of several result lists accumulates a stronger
score.

Alternative queries were stored in a dictionary so that evaluation could
be repeated consistently.

### 7. Query expansion

A query-expansion prompt was tested. It generated additional synonyms,
related expressions, and alternative phrasings.

The experiment did not produce a consistent retrieval improvement and
sometimes worsened ranking or distances. Query expansion was therefore
not included in the later formal comparison.

### 8. Contextual retrieval

A contextualisation stage was implemented during ingestion.

The process uses document-level analysis as reference material and asks
an LLM to add a short explanation to a chunk when the chunk needs
additional context to be understood independently.

The intended design is:

-   Preserve the original chunk text.
-   Store the added version separately as `contextualised_chunk`.
-   Keep the original chunk ID and metadata.
-   Avoid adding context to chunks that are already self-contained.
-   Add only information supported by the source document.
-   Embed and index the contextualised version for contextual retrieval.

Several prompt iterations were tested, including instructions to assess
whether context was necessary and to avoid redundant information. The
model sometimes still produced unnecessary repetition, so prompt tuning
was paused in favour of evaluating retrieval performance.

The corpus has been rebuilt with contextualised chunks and zero overlap
for this experiment.

**Important outstanding comparison:** the contextualised corpus has been
evaluated, but a complete, controlled baseline-versus-contextualised
comparison across the full test set is still required. Both collections
must use the same chunking, IDs, embedding model, queries, relevance
labels, and retrieval settings.

### 9. Cross-Encoder reranking

A Cross-Encoder reranking stage was implemented on top of vector
retrieval.

The current experiment:

1.  Retrieves 20 candidates from ChromaDB.
2.  Creates a query-document pair for each candidate.
3.  Scores all 20 pairs with the Cross-Encoder.
4.  Associates each score with the original chunk ID, document, and
    vector distance.
5.  Sorts candidates by Cross-Encoder score, highest first.
6.  Uses the top 10 reranked candidates for evaluation.

This is a two-stage retrieval design: vector search supplies the
candidate pool, and the Cross-Encoder reranks that pool.

A key limitation is that a Cross-Encoder cannot recover a relevant chunk
that is absent from the initial candidate pool.

------------------------------------------------------------------------

## Retrieval evaluation

### Test set

A manually labelled evaluation set was created with **20 questions**:

-   15 answerable questions about the two source documents.
-   5 unanswerable questions, including questions about facts not
    present in the documents and an unrelated general-knowledge
    question.

The ground-truth file maps each query to a list of relevant chunk IDs.
Empty lists identify questions that cannot be answered from the indexed
corpus.

The relevance labels are chunk-level: a chunk is relevant when it
contains useful information for answering the question, not merely
because it discusses the same general topic.

### Metrics

The evaluation uses Precision@K and Recall@K at:

$$
K \in \{1, 3, 5, 10\}
$$

**Precision@K**

$$
Precision@K = \frac{\text{Relevant chunks in top K}}{\text{Number of chunks retrieved in top K}}
$$

**Recall@K**

$$
Recall@K = \frac{\text{Relevant chunks in top K}}{\text{Total relevant chunks}}
$$

The reported averages are macro averages: each answerable query
contributes equally. The five unanswerable questions return `None` for
these metrics and are excluded from the averages. Zero scores for
answerable questions are retained.

This means the reported metrics measure retrieval quality for answerable
queries; they do not measure whether the system correctly refuses
unanswerable questions.

------------------------------------------------------------------------

## Experiments and findings

The following table summarises the macro-average results from the
contextualised corpus evaluation. Values are rounded to three decimal
places.

### Overall comparison

  ----------------------------------------------------------------------------------
  Metric                 K       Vector        Query   Cross-Encoder     Rewriting +
                                 search    rewriting                   Cross-Encoder
  ----------- ------------ ------------ ------------ --------------- ---------------
  Recall                 1        0.217    **0.250**           0.183           0.183

  Recall                 3        0.422    **0.511**           0.400           0.439

  Recall                 5        0.489    **0.600**           0.545           0.550

  Recall                10        0.644    **0.661**           0.639           0.639

  Precision              1        0.600    **0.667**           0.533           0.467

  Precision              3        0.422    **0.489**           0.377           0.400

  Precision              5        0.280    **0.347**           0.307           0.320

  Precision             10        0.187    **0.193**           0.187           0.187
  ----------------------------------------------------------------------------------

**Current finding:** query rewriting is the best-performing approach
among the four tested configurations on this evaluation set. It leads in
every reported precision and recall comparison.

### Individual experiment observations

#### Vector search

Vector search provides the baseline against which the other retrieval
approaches are compared.

It retrieves semantically similar chunks, but some exact technical
identifiers and specific concepts can be difficult to rank consistently.

#### Query rewriting

Query rewriting improved the macro-average results over ordinary vector
search at every evaluated K.

Its strongest results were:

-   Recall@3: 0.511
-   Recall@5: 0.600
-   Precision@3: 0.489
-   Precision@5: 0.347

It is the current best-performing retrieval approach in these
experiments.

#### Multi-query retrieval with RRF

The earlier multi-query experiment showed query-dependent improvements.
It improved retrieval for some questions, particularly at certain top-K
cutoffs, but did not consistently outperform ordinary retrieval or query
rewriting.

In the earlier three-query TurboQuant experiment, the macro averages
were:

  Metric         K   Normal   Rewriting   Multi-query + RRF
  ----------- ---- -------- ----------- -------------------
  Recall         1     0.25        0.25                0.25
  Recall         3     0.42        0.50                0.50
  Recall         5     0.50        0.50                0.58
  Recall        10     0.58        0.58                0.58
  Precision      1     0.67        0.67                0.67
  Precision      3     0.33        0.44                0.44
  Precision      5     0.27        0.27                0.33
  Precision     10     0.17        0.17                0.17

This was a small three-query experiment and should not be treated as
directly comparable to the later 15-query evaluation.

#### Query expansion

Query expansion was tested but did not show a reliable improvement. It
was set aside rather than included in the final comparison.

#### Contextual retrieval

Contextualised chunks were successfully generated, embedded, stored, and
evaluated. The current comparison table reflects retrieval over the
contextualised corpus.

A controlled evaluation of the contextualised corpus against the
original non-contextualised corpus remains outstanding. Therefore, the
current results do not establish the amount of improvement attributable
specifically to contextualisation.

#### Cross-Encoder reranking

Cross-Encoder reranking improved some results relative to vector search,
especially at K=5:

-   Recall@5: 0.489 → 0.545
-   Precision@5: 0.280 → 0.307

However, it did not improve every ranking cutoff. It also performed
below query rewriting alone.

#### Query rewriting + Cross-Encoder

Combining query rewriting with Cross-Encoder reranking improved some
metrics over Cross-Encoder reranking alone:

-   Recall@3: 0.400 → 0.439
-   Recall@5: 0.545 → 0.550
-   Precision@3: 0.377 → 0.400
-   Precision@5: 0.307 → 0.320

However, it did not outperform query rewriting alone. The combined
pipeline therefore adds complexity without demonstrating a net
improvement on this test set.

------------------------------------------------------------------------

## Current best-performing approach

Based on the present evaluation:

**Query rewriting followed by vector retrieval is the current preferred
retrieval configuration.**

This is an empirical conclusion for the current corpus and test set, not
a claim that query rewriting is universally superior.

The Cross-Encoder implementation remains useful as a completed
experiment and may be revisited if candidate selection, model choice, or
reranking configuration changes.

------------------------------------------------------------------------

## Limitations and open questions

1.  **Small corpus and test set:** the evaluation contains two source
    documents and 15 answerable queries. Results may not generalise to
    larger or more varied knowledge bases.
2.  **Manual relevance labels:** ground truth is manually authored.
    Labels should be reviewed as the corpus and chunking strategy
    evolve.
3.  **Contextual retrieval comparison:** a controlled
    baseline-versus-contextualised comparison across all queries is
    still needed.
4.  **Unanswerable questions:** empty ground-truth lists are excluded
    from Precision@K and Recall@K. A separate abstention evaluation is
    needed.
5.  **Candidate-pool ceiling:** Cross-Encoder reranking cannot recover
    relevant documents omitted by initial vector retrieval.
6.  **Distance threshold:** the initial RAG pipeline used a threshold of
    0.8. It should be calibrated against answerable and unanswerable
    queries rather than assumed to be optimal.
7.  **Generation quality:** the current reported metrics assess
    retrieval, not faithfulness, answer correctness, citation quality,
    or hallucination rate.
8.  **Runtime and cost:** latency and token usage have not yet been
    systematically compared across the retrieval configurations.

------------------------------------------------------------------------

## Roadmap

### Completed / implemented

-   [x] Basic document ingestion and loading
-   [x] Sentence-aware chunking
-   [x] Separate embedding-generation stage
-   [x] ChromaDB storage and vector retrieval
-   [x] Context construction and LLM generation
-   [x] Precision@K and Recall@K evaluation functions
-   [x] Manually labelled ground-truth dataset
-   [x] Query rewriting
-   [x] Multi-query retrieval
-   [x] Reciprocal Rank Fusion
-   [x] Query expansion experiment
-   [x] Contextual chunk generation
-   [x] Cross-Encoder reranking
-   [x] Evaluation of query rewriting + Cross-Encoder combination

### Next: retrieval optimisation

-   [ ] Run a controlled baseline-versus-contextualised corpus
    comparison.
-   [ ] Implement BM25 independently.
-   [ ] Evaluate BM25 against the existing vector-search baseline.
-   [ ] Combine BM25 and vector search using rank fusion.
-   [ ] Compare hybrid retrieval with query rewriting.
-   [ ] Investigate metadata filtering.
-   [ ] Explore parent-child retrieval and query decomposition.
-   [ ] Explore HyDE and multi-vector retrieval where useful.
-   [ ] Revisit candidate-pool size and final top-K selection.

### Generation quality and RAG evaluation

-   [ ] Improve prompt design and grounded generation.
-   [ ] Evaluate answer correctness and faithfulness.
-   [ ] Add source attribution and citation checks.
-   [ ] Implement and evaluate abstention for unanswerable questions.
-   [ ] Evaluate context precision and context recall.
-   [ ] Track latency and token usage.
-   [ ] Explore RAGAS or other systematic evaluation methods.

### Advanced architectures

-   [ ] Agentic RAG
-   [ ] Corrective RAG
-   [ ] Self-RAG
-   [ ] Graph RAG
-   [ ] Adaptive RAG
-   [ ] Multi-hop retrieval

### Production engineering

-   [ ] Expose the pipeline through FastAPI.
-   [ ] Add asynchronous ingestion and retrieval where appropriate.
-   [ ] Add background document-processing jobs.
-   [ ] Explore alternative vector databases.
-   [ ] Add caching and conversation memory.
-   [ ] Containerise and deploy the application.
-   [ ] Add authentication and operational monitoring.

------------------------------------------------------------------------

## Learning approach

This project is being developed as a learning exercise, not merely
assembled from a framework.

For each technique, the intended process is:

1.  Understand what the technique does.
2.  Understand why it is useful and what trade-offs it introduces.
3.  Consider alternative approaches and their advantages or
    disadvantages.
4.  Implement the technique independently.
5.  Evaluate it using the same corpus and labelled questions.
6.  Compare the measured results before deciding whether to retain it.

The guiding principle is to prefer measured improvements over adding
complexity for its own sake.
