"""CLI for RAG ML Docs.

Usage:
  ~/workspace/ml_env/bin/python ask.py "How do I use StandardScaler?"
  ~/workspace/ml_env/bin/python ask.py --generative "...."
  ~/workspace/ml_env/bin/python ask.py --debug "...."
  ~/workspace/ml_env/bin/python ask.py            # interactive, keeps history
"""

import argparse
import os
import sys

from rag_pipeline import RAGIndex, ChatSession, answer_question


def render(result):
    print(f"\nQ: {result['question']}")
    if len(result["sub_questions"]) > 1:
        print("Decomposed into:")
        for sq in result["sub_questions"]:
            print(f"  - {sq}")
    if result["used_history"]:
        print("(used conversation history for this follow-up)")
    print(f"Mode: {result['mode']}")
    if result["generative_text"]:
        print(f"\n{result['generative_text']}\n")
    if not result["citations"]:
        print("No relevant passages found in the indexed docs.")
        return
    print("Sources:")
    for c in result["citations"]:
        print(f"\n[{c['rank']}] {c['title']} — {c['section']}")
        print(f"    {c['url']}")
        print(f"    {c['text'][:500]}")
    print(f"\nCitations verified: {result['citations_verified']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--generative", action="store_true",
                    help="try flan-t5-small generation (quality-gated)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    print("Loading index...", file=sys.stderr)
    index = RAGIndex()
    session = ChatSession()
    generative = args.generative or bool(os.environ.get("RAG_GENERATIVE"))

    questions = [" ".join(args.question)] if args.question else []
    if not questions:
        print("Interactive mode (blank line quits, history kept):")
        while True:
            try:
                q = input("\nAsk: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            render(answer_question(index, q, session=session,
                                   generative=generative,
                                   debug=args.debug))
    else:
        for q in questions:
            render(answer_question(index, q, session=session,
                                   generative=generative,
                                   debug=args.debug))


if __name__ == "__main__":
    main()
