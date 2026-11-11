# CosmoCite: Project Report

The reported results are drawn from the evaluation files in `reports/`. Evaluation methods and the corresponding results are documented alongside each experiment.

## 1. Problem

Researchers working in space biology often need to investigate questions across hundreds of scientific publications and trace each finding back to its original source. General-purpose chatbots can produce fluent answers without reliable evidence, making it difficult to verify scientific claims.

CosmoCite addresses this problem through evidence-grounded question answering. It retrieves passages from scientific publications, selects relevant evidence, links answer sentences to their sources, and abstains when the available evidence is insufficient.

Its central objective is to make scientific information easier to retrieve, verify, and explore through direct links to published research.

## 2. System

**Corpus.** The collection contains 607 PMC links obtained from the NASA `SB_publications` list. Of these, 572 were successfully parsed, producing 20,242 passages of up to 180 words each. Dataset splits are performed at the paper level rather than the passage level, ensuring that passages from the same paper do not appear across different splits.

**Retrieval and answering pipeline.**

1. **Retrieve:** BM25 and dense vector retrieval identify candidate passages. Their rankings are combined using reciprocal rank fusion.
2. **Rerank:** A cross-encoder re-scores the top 50 retrieved passages.
3. **Check evidence:** The system refuses to answer if the highest passage score is below 4.1. This threshold was tuned using the development split.
4. **Filter sentences:** Candidate sentences are selected based on their relevance to the question. The filter prioritises findings about the requested tissue and stressor while excluding aims, methods, open questions, and generic descriptions.
5. **Rank sentences:** The cross-encoder scores the remaining sentences, restricted to lengths of 8–60 words.
6. **Select evidence:** The highest-ranked sentence is selected, with up to two additional sentences from other papers when their scores meet the threshold of 3.0.
7. **Prefer the requested organism:** Evidence concerning the organism specified in the question is prioritised whenever it meets the selection threshold.
8. **Cite sources:** Each selected sentence links to its original passage and PMC publication. Publication authors, year, and journal metadata are obtained from NCBI.

**Optional generation engine.** An alternative answering mode sends retrieved passages to a locally hosted Ollama model (`qwen2.5:0.5b`). Generated sentences are checked against their source passages to maintain consistency with the retrieved evidence. FLAN-T5 is retained for comparison between decoding approaches.

**Design decision: evidence checking precedes answering.** The evidence threshold is applied before the remaining answering stages, ensuring that every answering method follows the same evidence requirements.

## 3. Evaluation

### 3.1 Faithfulness and relevance

**Source faithfulness.** The default sentence-ranking engine selects sentences directly from retrieved passages. All 60 sampled output sentences were found word for word in their corresponding passages, confirming that the selected quotations exist in their cited sources.

**Topical relevance.** Relevance was assessed using a consistent criterion: whether a sentence reports an effect of the requested stressor on the requested tissue in the requested organism. The labels were assigned by an AI model (Claude Opus 5.5), which was also used to build the filter; they are not human labels. The assessment results are recorded in `reports/relevance_check.json`.

| Evaluation stage | Sentences assessed | On-topic sentences |
|---|---:|---:|
| Before filtering | 60 | 38 (63%) |
| After filtering and organism preference | 62 | 51 (82%) |

The results show an increase in the proportion of selected sentences matching the requested topic after applying sentence filtering and organism preference. Two cautions apply: the before and after samples differ, and the filter was developed on these same questions, so the figures are a rough estimate rather than a held-out measurement.

The filtering process was also evaluated on naturally phrased questions. Initial testing on 16 natural questions informed improvements to the matching rules, including recognition of expressions such as "visual" and "in-flight".

A further evaluation on 14 fresh natural questions produced 10 identical answers and four minor changes, without introducing additional refusals.

A set of 30 natural questions is retained as a regression check. The default system answers 28 and refuses two. These results are recorded in `reports/natural.json`.

### 3.2 Retrieval performance

The retrieval experiments compare BM25, dense retrieval, hybrid retrieval, and cross-encoder reranking.

| Evaluation set | Metric | BM25 | Dense | Hybrid | Reranked |
|---|---|---:|---:|---:|---:|
| Judged questions (12) | nDCG@5 | 0.252 | 0.817 | 0.623 | 0.554 |
| Known-item questions (30) | MRR | 0.891 | 0.811 | 0.848 | 0.903 |

Dense retrieval achieves the highest nDCG@5 on the judged retrieval questions, while reranking achieves the highest MRR on the known-item set. The relevance labels for the 12 judged questions were assigned by an AI model (Claude Opus 5.5) from paper titles and abstracts, blind to the retrieval mode; they are not human labels, and 12 questions is a small sample.

The current system retains reranking as its default retrieval strategy even though dense retrieval scores higher on the judged set. The evidence check and the sentence ranking both rely on cross-encoder scores, and the refusal threshold of 4.1 is calibrated on them, so reranking keeps one scoring scale across the whole answering pipeline. The gap on the judged set is a known trade-off of this choice.

### 3.3 Model experiments

The project also includes experiments involving scientific text classification, named entity recognition, and sequence-to-sequence models.

| Task | Baseline | Improved model | Final model |
|---|---|---|---|
| Section classifier (macro-F1) | TF-IDF: 0.654 | BiLSTM: 0.616 | SciBERT: 0.691 |
| NER (entity F1) | Rules: 0.650 | n/a | BioBERT: 0.626 |
| Seq2seq (ROUGE-L) | n/a | Bahdanau: 0.127, Luong: 0.154 | T5: code in `seq2seq.py`, not evaluated |

These experiments document the project's exploration of different machine-learning approaches for scientific text processing.

## 4. Path to Scale

The current architecture keeps dense embeddings and the BM25 index in memory. This is suitable for the existing collection of approximately 20,000 passages.

**Implemented improvements.**

1. **Indexing:** Dense retrieval is accessed through a common interface, with exact NumPy search as the default and FAISS HNSW available as an alternative. Retrieval fusion operates on the top 1,000 candidates from each retriever. At 1,000 candidates, all retrieval metrics were within 0.0001 of full-passage ranking for both index types on the silver-labelled questions.
2. **Incremental ingestion:** `ingest.py` supports adding papers without rebuilding the entire corpus. Existing identifiers and dataset splits remain fixed. A paper removed from a copy of the corpus was successfully ingested again, restoring all 51 passages unchanged.
3. **Flexible input sources:** `fetch.py --csv` supports fetching publications from different PMC lists.

**Measured performance.**

The following measurements are recorded in `reports/scale.json`. They were collected on a single laptop using 1,000 results per search. Random-data measurements use 200 random queries, while real-data measurements use the evaluation questions.

| Vectors | Data | Index | Build (s) | p50 (ms) | p95 (ms) | Recall@10 |
|---:|---|---|---:|---:|---:|---:|
| 20,242 | Real | NumPy | 0.00 | 0.84 | 1.06 | 1.000 |
| 20,242 | Real | FAISS | 0.51 | 3.69 | 8.39 | 1.000 |
| 20,242 | Random | NumPy | 0.00 | 0.74 | 1.00 | 1.000 |
| 20,242 | Random | FAISS | 0.72 | 5.98 | 9.90 | 0.996 |
| 200,000 | Random | NumPy | 0.00 | 6.70 | 7.84 | 1.000 |
| 200,000 | Random | FAISS | 19.03 | 11.80 | 17.79 | 0.581 |

Exact NumPy search performed better than FAISS in the reported experiments, so it remains the default indexing method.

At 200,000 vectors, the experiment used random data, while real scientific embeddings were available for the 20,242-passage collection. These measurements provide a basis for comparing indexing approaches at different collection sizes.

The benchmark script also lists a 500,000-vector size, but that run did not complete and no result was recorded. These measurements cover vector-search performance rather than the complete question-answering pipeline. A full answer takes approximately one second in the reported setup, with most of the time spent in cross-encoder scoring.

**Additional engineering decisions.**

- **Keyword indexing:** BM25 is rebuilt at startup. SQLite FTS5 was considered, but the existing startup time of approximately 10–25 seconds was considered suitable for the current corpus size.
- **Verification:** NLI-based semantic verification remains an area for future development.

## 5. Conclusion

CosmoCite brings together scientific literature retrieval, dense and lexical search, cross-encoder reranking, sentence-level evidence selection, and direct source citations in a unified question-answering system. The project combines a collection of 20,242 scientific passages with a multi-stage retrieval and answering pipeline. Its experiments demonstrate the use of sentence filtering, organism-aware evidence selection, and source-linked answers. The system also includes incremental ingestion, configurable retrieval methods, local model integration, and performance measurements across different vector collection sizes. CosmoCite provides a practical foundation for exploring and retrieving evidence from space-biology publications, making scientific findings more accessible and easier to trace back to their original research sources.
