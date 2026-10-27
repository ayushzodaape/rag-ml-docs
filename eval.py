"""Eval harness for RAG ML Docs.

10 fixed questions, each with expected source doc(s). Measures, honestly:
  - hit@5: an expected doc appears among the top-5 reranked citations
  - hit@10: an expected doc appears among the fused top-10 candidates
  - top1: the top-1 citation is an expected doc
  - citations_verified: every citation resolves to a stored chunk
  - decomposition: the multi-part question splits into >1 sub-question

Usage:  ~/workspace/ml_env/bin/python eval.py
Output: printed table + eval_results.json
"""

import json
from pathlib import Path

from rag_pipeline import RAGIndex, answer_question

BASE = Path(__file__).resolve().parent

QUESTIONS = [
    ("How do I scale features with StandardScaler before training a model?",
     {"sklearn-scaler"}),
    ("How do I build a Pipeline that combines a scaler and a classifier?",
     {"sklearn-pipeline"}),
    ("How do I tune hyperparameters with GridSearchCV?",
     {"sklearn-gridsearch"}),
    ("How do I evaluate a classifier with a classification report, and "
     "what do precision and recall mean?",
     {"sklearn-classification-report"}),
    ("How do I create a DataLoader for a custom Dataset in PyTorch?",
     {"torch-data"}),
    ("How do I define a custom neural network by subclassing nn.Module?",
     {"torch-module"}),
    ("How does autograd compute gradients, and what does calling "
     "backward() do?", {"torch-autograd"}),
    ("How do I use the Adam optimizer and set the learning rate?",
     {"torch-adam"}),
    ("How do I evaluate a model with cross_val_score?",
     {"sklearn-cross-val"}),
    # multi-part: exercises query decomposition
    ("How do I scale features with StandardScaler and then train a "
     "LogisticRegression model in a Pipeline?",
     {"sklearn-scaler", "sklearn-logreg", "sklearn-pipeline"}),
]


def main():
    index = RAGIndex()
    rows = []
    for i, (q, expected) in enumerate(QUESTIONS, 1):
        res = answer_question(index, q, debug=True)
        top5_docs = {c["doc_id"] for c in res["citations"]}
        fused_docs = {index.chunks[c["id"]]["doc_id"]
                      for c in res["debug"]["fused_candidates"]}
        top1_doc = res["citations"][0]["doc_id"] if res["citations"] else None
        row = {
            "n": i,
            "question": q,
            "expected": sorted(expected),
            "top1_doc": top1_doc,
            "top5_docs": sorted(top5_docs),
            "hit@5": bool(top5_docs & expected),
            "hit@10": bool(fused_docs & expected),
            "top1": top1_doc in expected,
            "citations_verified": res["citations_verified"],
            "n_sub_questions": len(res["sub_questions"]),
            "sub_questions": res["sub_questions"],
            "mode": res["mode"],
        }
        rows.append(row)
        print(f"Q{i}: hit@5={row['hit@5']} hit@10={row['hit@10']} "
              f"top1={row['top1']} verified={row['citations_verified']} "
              f"sub_q={row['n_sub_questions']} top1_doc={top1_doc}")

    n = len(rows)
    summary = {
        "n_questions": n,
        "hit@5": sum(r["hit@5"] for r in rows) / n,
        "hit@10": sum(r["hit@10"] for r in rows) / n,
        "top1": sum(r["top1"] for r in rows) / n,
        "citations_verified_all": all(r["citations_verified"] for r in rows),
        "decomposition_triggered_q10": rows[9]["n_sub_questions"] > 1,
    }
    print(f"\nSUMMARY: hit@5={summary['hit@5']:.2f} "
          f"hit@10={summary['hit@10']:.2f} top1={summary['top1']:.2f} "
          f"verified_all={summary['citations_verified_all']} "
          f"decomp_q10={summary['decomposition_triggered_q10']}")
    (BASE / "eval_results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1),
        encoding="utf-8")


if __name__ == "__main__":
    main()
