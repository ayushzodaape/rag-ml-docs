# Demo — measured outcomes

All numbers below were measured on 2026-10-27 on CPU, via `eval.py`
(fixed 10 questions, known expected docs). `hit@5` = expected doc among
the top-5 cited passages; `top1` = expected doc is the #1 passage;
`verified` = every citation resolves to a stored chunk.
Full machine-readable output: `eval_results.json`.

## Summary

| Metric | Result |
|---|---|
| hit@5 | 0.90 (9/10) |
| hit@10 | 1.00 (10/10) |
| top-1 | 0.80 (8/10) |
| citations verified | 10/10 questions |

## Per-question results

| # | Question | Sub-q | Expected doc | Top-1 doc | hit@5 | hit@10 | top1 | verified |
|---|---|---|---|---|---|---|---|---|
| Q1 | How do I scale features with StandardScaler? | 1 | sklearn-scaler | sklearn-scaler | ✓ | ✓ | ✓ | ✓ |
| Q2 | How do I build a Pipeline with a scaler and a classifier? | 1 | sklearn-pipeline | sklearn-pipeline | ✓ | ✓ | ✓ | ✓ |
| Q3 | How do I tune hyperparameters with GridSearchCV? | 1 | sklearn-gridsearch | sklearn-svc | ✓ | ✓ | ✗ | ✓ |
| Q4 | How do I evaluate a classifier and print precision and recall? | 2 | sklearn-classification-report | sklearn-classification-report | ✓ | ✓ | ✓ | ✓ |
| Q5 | How do I load data with a DataLoader? | 1 | torch-data | torch-data | ✓ | ✓ | ✓ | ✓ |
| Q6 | How do I define a custom neural network module? | 1 | torch-module | torch-module | ✓ | ✓ | ✓ | ✓ |
| Q7 | How does autograd compute gradients and how do I disable it? | 2 | torch-autograd | torch-autograd | ✓ | ✓ | ✓ | ✓ |
| Q8 | How do I use the Adam optimizer and set the learning rate? | 2 | torch-adam | torch-steplr | ✗ | ✓ | ✗ | ✓ |
| Q9 | How do I cross-validate a model? | 1 | sklearn-cross-val | sklearn-cross-val | ✓ | ✓ | ✓ | ✓ |
| Q10 | How do I scale features with StandardScaler and then train a LogisticRegression model in a Pipeline? | 2 | scaler / logreg / pipeline | sklearn-scaler | ✓ | ✓ | ✓ | ✓ |

## The two misses, honestly

- **Q8 (hit@5 miss):** the compound phrasing "…and set the learning
  rate" pulls scheduler content; the reranker filled the entire top-5
  with StepLR chunks, pushing Adam out (Adam is still in the fused
  top-10, so hit@10 holds). Counter-check: the plain question
  "How do I use the Adam optimizer?" retrieves torch.optim.Adam in all
  5 top slots via the Flask `/ask` endpoint — so the miss is a reranker
  preference on compound phrasing, not a missing doc.
- **Q3 (top-1 miss only):** GridSearchCV is in the top-5, but the SVC
  page outranks it at #1 because that page shows GridSearchCV examples.

## Decomposition demo

Q10 decomposes into exactly 2 sub-questions, retrieved per sub-question,
then synthesized:

```
Q: How do I scale features with StandardScaler and then train a
   LogisticRegression model in a Pipeline?
Decomposed into:
  - How do I scale features with StandardScaler
  - train a LogisticRegression model in a Pipeline
Mode: extractive
Sources: sklearn.preprocessing.StandardScaler, sklearn.pipeline.Pipeline
Citations verified: True
```

## Generative mode demo — 0/3 usable (extractive is the recommendation)

With `RAG_GENERATIVE=1`, flan-t5-small produced, on three demo questions:

1. a signature line (`class sklearn.preprocessing.StandardScaler…`),
2. a bare type list (`float, int, bool…`),
3. a raw code snippet.

All three were correctly rejected by the quality gate (must end with
sentence punctuation, no `>>>` artefacts, no bare type fragments), and
the system fell back to the extractive answer. The CLI reports this as
`Mode: extractive (generative rejected)`. **Do not claim fluent
generated answers for this project.**

## Conversation memory demo

A follow-up such as "and what about the learning rate?" after an Adam
question is expanded with the previous question before retrieval; a
short standalone question ("How do I scale features?") is not expanded.
The expansion is visible in the CLI/Flask response as `sub_questions`.

## How to reproduce

```bash
python eval.py          # prints the per-question table + SUMMARY line
python ask.py "How do I scale features with StandardScaler and then train a LogisticRegression model in a Pipeline?"
RAG_GENERATIVE=1 python ask.py "How do I scale features with StandardScaler?"
python app.py           # then POST {"question": "..."} to /ask
```
