# RAG over ML Documentation and Code Examples

Answers machine-learning questions grounded in **official ML library
documentation** — scikit-learn and PyTorch — with code examples taken from
those pages. This is a different genre from the companion project
`rag-paper-explorer` (which answers over research papers): here every answer
traces to an official docs page, with the doc title, section, and URL cited.

**Live demo: https://rag-ml-docs.onrender.com** (free tier — spins down when idle, first load takes ~a minute). No API keys, no paid services.

## Architecture

```
Question
  └─ 1. Query decomposition (heuristic, rule-based — NOT LLM decomposition)
       splits a compound question on "and then"/"and"/"with" into sub-questions
  └─ 2. Hybrid retrieval per sub-question
       BM25 (pure Python) + dense embeddings fused 50/50 after min-max
       normalisation → fused top-10 per sub-question
  └─ 3. Cross-encoder reranking
       cross-encoder/ms-marco-MiniLM-L-6-v2 over the fused top-30 → top-5
  └─ 4. Citation verification
       every cited passage must resolve to a stored chunk (doc + section);
       the answer is EXTRACTIVE by default: the top passages themselves,
       each labelled with its source
  └─ 5. Generative mode (OPT-IN, gated, NOT recommended)
       flan-t5-small behind RAG_GENERATIVE=1. A quality gate rejects
       outputs that do not end with sentence punctuation, contain ">>>"
       artefacts, or are bare type fragments — on rejection the system
       falls back to the extractive answer. See "Honest limitations".
  └─ 6. Conversation memory
       follow-up questions ("and what about…", "how does that work?",
       pronouns) are expanded with the previous question before retrieval;
       short standalone questions are NOT expanded.
```

Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (CPU, cached locally).
Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU).
Storage: FAISS index (`data/index.faiss`) + chunk store (`data/chunks.json`).

## Corpus

**43 pages, all from official documentation only — 0 skipped.**
23 scikit-learn + 20 PyTorch. Fetched 2026-10-27 from
`scikit-learn.org` and `pytorch.org/docs` (PyTorch `/stable/` URLs serve
JS-redirect stubs, so the fetcher follows them to the versioned `/2.14/`
pages; the stable URL is kept as the citation URL).

Coverage: sklearn estimators (LogisticRegression, LinearRegression, SVC,
RandomForest, HistGradientBoosting, KNN, KMeans, DecisionTree, GaussianNB),
model selection (GridSearchCV, cross_val_score, train_test_split),
preprocessing (StandardScaler, OneHotEncoder, SimpleImputer, PCA,
SelectKBest), pipelines (Pipeline, ColumnTransformer), metrics
(accuracy_score, classification_report, confusion_matrix, roc_auc_score);
PyTorch tensors (tensor, manual_seed), nn module (Module, Sequential,
Linear, Conv2d, ReLU, Softmax, Dropout, BatchNorm1d, CrossEntropyLoss,
MSELoss), optimizers (Adam, SGD, StepLR), autograd (autograd notes,
no_grad), data loading (data.html — DataLoader/Dataset), save/load.

Chunking: 900 chars with 150 overlap → **1,954 chunks**.

### Corpus source URLs

scikit-learn (23):

- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
- https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
- https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
- https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

PyTorch (20):

- https://docs.pytorch.org/docs/stable/generated/torch.tensor.html
- https://docs.pytorch.org/docs/stable/generated/torch.manual_seed.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
- https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
- https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html
- https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
- https://docs.pytorch.org/docs/stable/notes/autograd.html
- https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html
- https://docs.pytorch.org/docs/stable/data.html
- https://docs.pytorch.org/docs/stable/generated/torch.save.html
- https://docs.pytorch.org/docs/stable/generated/torch.load.html

If a page cannot be fetched from the official site, it is skipped — no
substitute source is ever used.

## How to run

```bash
pip install -r requirements.txt

# 1. Fetch the corpus (official docs only)
python fetch_docs.py

# 2. Chunk + build the FAISS index
python build_index.py

# 3a. Ask from the CLI (extractive by default)
python ask.py "How do I scale features with StandardScaler?"

# 3b. Generative mode (opt-in; falls back to extractive on quality failure)
RAG_GENERATIVE=1 python ask.py "How do I scale features with StandardScaler?"

# 3c. Web UI
python app.py          # http://127.0.0.1:5000  (local only)
```

The CLI prints the decomposition, the mode, the top-5 cited passages
(doc title, section, URL, excerpt), and whether citations verified.
The Flask `/ask` endpoint returns the same as JSON.

## Verified results (measured 2026-10-27, CPU)

Eval harness: `eval.py` — 10 fixed questions with known expected source
docs. Full per-question table in `DEMO.md`; machine-readable results in
`eval_results.json`.

| Metric | Result |
|---|---|
| Retrieval hit@5 (expected doc in top-5) | **0.90 (9/10)** |
| Retrieval hit@10 | **1.00 (10/10)** |
| Top-1 correct | **0.80 (8/10)** |
| Citations verified (all 10 questions) | **True** |
| Decomposition triggered on the multi-part question | **Yes (2 sub-questions)** |

Also verified end-to-end by hand: Flask homepage serves; Flask `/ask`
returns Adam citations for an Adam question; CLI answers a compound
Scaler + Pipeline question with the right docs; generative flag prints
"extractive (generative rejected)" and falls back cleanly.

## Honest limitations — read before claiming anything

- **Generative mode is not usable: 0/3 demo generations passed the
  quality gate.** flan-t5-small produced a signature line, a bare type
  list, and a code snippet — all correctly rejected. Do NOT claim fluent
  generated answers; extractive is the only recommended mode.
- **One retrieval miss: Q8 "How do I use the Adam optimizer and set the
  learning rate?"** — the expected Adam doc is in the fused top-10 but
  the reranker filled the top-5 entirely with StepLR chunks (the
  "set the learning rate" half pulls scheduler content). The plain
  single-topic Adam question retrieves Adam in all 5 slots, so this is a
  reranker preference on compound phrasing, not a corpus gap.
- **Q3 top-1 miss:** GridSearchCV is in the top-5, but the SVC page
  (which shows GridSearchCV examples) outranks it at top-1.
- **Query decomposition is heuristic**, not LLM-based: it splits only
  what the user wrote, on fixed connectors. It will not handle
  genuinely ambiguous phrasing.
- **Page truncation:** per-page text is capped during fetch
  (nn.Module's page is long; the cap keeps the index build tractable).
- No OCR, no non-official sources, no deployment. The corpus is the 43
  pages above — questions outside it will retrieve the nearest docs,
  which may be the wrong ones, and the answer will say so only via its
  citations.

## Files

| File | Purpose |
|---|---|
| `fetch_docs.py` | Fetch official docs pages, follow PyTorch redirects, detect stubs |
| `build_index.py` | Chunk corpus, embed, build FAISS index |
| `rag_pipeline.py` | Decomposition, hybrid retrieval, reranking, citation verification, generative gate, conversation memory |
| `ask.py` | CLI |
| `app.py` + `templates/index.html` | Flask web UI |
| `eval.py` | Fixed 10-question eval harness |
| `corpus/docs.json`, `corpus/fetch_log.json` | Fetched corpus + fetch log |
| `data/chunks.json`, `data/index.faiss` | Chunk store + FAISS index |
| `eval_results.json` | Measured eval output |
| `requirements.txt` | Dependencies |
