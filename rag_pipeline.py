"""RAG ML Docs — core pipeline.

Answers ML questions grounded in official scikit-learn and PyTorch
documentation. Retrieval stack, in order:

1. Query decomposition — a compound question is split heuristically into
   sub-questions (sentence boundaries, semicolons, numbered parts, and
   " and " joins where the second half starts a new clause). This is a
   transparent heuristic, NOT an LLM: it never invents sub-questions, it
   only splits what the user wrote.
2. Hybrid retrieval — BM25 (pure-Python implementation) + dense embeddings
   (sentence-transformers/all-MiniLM-L6-v2, CPU), fused 50/50 after
   min-max normalisation over the candidate set.
3. Cross-encoder reranking — cross-encoder/ms-marco-MiniLM-L-6-v2 over the
   fused top-30, keeping the top-5.
4. Extractive answers by default — the answer IS the retrieved passages,
   each cited with doc title, section and URL. Citation verification checks
   every citation resolves to a stored chunk.
5. Generative mode (flan-t5-small) is available ONLY behind
   RAG_GENERATIVE=1 and only passes a quality gate (content-word overlap
   with the retrieved passages); otherwise it falls back to extractive.
6. Conversation memory — a ChatSession keeps exchanges; short follow-ups
   ("and what about...", "why?") expand retrieval with the previous
   question for context.

No API keys, no paid services, no logins. Everything runs on CPU.
"""

import json
import math
import os
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
CORPUS_DIR = BASE / "corpus"
DATA_DIR = BASE / "data"
DOCS_PATH = CORPUS_DIR / "docs.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"
INDEX_PATH = DATA_DIR / "index.faiss"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEN_MODEL = "google/flan-t5-small"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
DENSE_K = 25
BM25_K = 25
FUSED_K = 30
RERANK_K = 5

STOPWORDS = set(
    "the a an and or of to in is are was were be been it its this that "
    "those these with for on by from as at how do does did i you we what "
    "when where which who whom why can could should would is it".split()
)

_embedder = None
_reranker = None
_generator = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def _get_generator():
    # transformers 5.x removed the text2text-generation pipeline task, so
    # load tokenizer + seq2seq model directly (same pattern the first RAG
    # project uses).
    global _generator
    if _generator is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(GEN_MODEL)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
        _generator = (tok, mdl)
    return _generator


# ---------------------------------------------------------------- chunking

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks, preferring sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sent in sentences:
        # hard-split a single oversized sentence (e.g. long code lines)
        while len(sent) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sent[:chunk_size])
            sent = sent[chunk_size:]
        if len(current) + len(sent) + 1 <= chunk_size:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + " " + sent).strip() if tail else sent
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) > 40]


def build_chunks(docs):
    """Flatten docs.json sections into chunk dicts with source metadata."""
    chunks = []
    for doc in docs:
        for sec in doc["sections"]:
            for text in chunk_text(sec["text"]):
                chunks.append({
                    "id": len(chunks),
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "source": doc["source"],
                    "section": sec["heading"],
                    "text": text,
                })
    return chunks


def build_index(chunks):
    """Embed chunks and write data/chunks.json + data/index.faiss."""
    import faiss
    import numpy as np

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    texts = [c["text"] for c in chunks]
    emb = _get_embedder().encode(texts, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))
    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=1), encoding="utf-8")
    return len(chunks)


# ---------------------------------------------------------------- BM25

def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in STOPWORDS]


class BM25:
    """Small pure-Python BM25 (k1=1.5, b=0.75) over the chunk corpus."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.docs = [tokenize(c["text"]) for c in chunks]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.df = {}
        for d in self.docs:
            for term in set(d):
                self.df[term] = self.df.get(term, 0) + 1
        self.n = len(self.docs)

    def scores(self, query):
        import numpy as np

        qterms = tokenize(query)
        out = np.zeros(self.n, dtype=float)
        k1, b = 1.5, 0.75
        for term in set(qterms):
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, doc in enumerate(self.docs):
                f = doc.count(term)
                if f:
                    out[i] += idf * (f * (k1 + 1)) / (
                        f + k1 * (1 - b + b * self.doc_len[i] / self.avgdl))
        return out


# ---------------------------------------------------------------- decomposition

_CLAUSE_VERBS = ("use", "create", "train", "build", "save", "load", "evaluate",
                 "compute", "apply", "define", "implement", "tune", "scale",
                 "convert", "fit", "predict", "plot", "choose", "set", "get",
                 "write", "run", "test", "handle", "split", "encode")


def decompose_query(question):
    """Split a compound question into sub-questions (heuristic, no LLM).

    Splits on sentence boundaries, semicolons, numbered parts, and on
    ' and ' when the second half starts a new clause (question word or
    clause verb). Returns the fragments; if nothing splits, returns the
    original question alone.
    """
    q = re.sub(r"\s+", " ", question).strip()
    parts = re.split(r"\s*;\s*|\s+(?:and then|then|also)\s+"
                     r"|\s*\d+[.)]\s+|(?<=[.?])\s+", q)
    fragments = []
    for part in parts:
        part = part.strip(" .?")
        if not part:
            continue
        m = re.split(r"\s+and\s+", part, maxsplit=1)
        if len(m) == 2 and len(m[0]) >= 15 and len(m[1]) >= 15:
            second = m[1].lower()
            if second.startswith(("how", "what", "why", "when", "which")) \
                    or second.split()[0] in _CLAUSE_VERBS:
                fragments.extend([m[0].strip(), m[1].strip()])
                continue
        fragments.append(part)
    # de-duplicate, keep order, drop tiny fragments
    seen, out = set(), []
    for frag in fragments:
        key = frag.lower()
        if len(frag) >= 10 and key not in seen:
            seen.add(key)
            out.append(frag)
    return out or [q]


# ---------------------------------------------------------------- retrieval

class RAGIndex:
    def __init__(self):
        import faiss

        self.chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(INDEX_PATH))
        self.bm25 = BM25(self.chunks)

    def dense_search(self, query, k=DENSE_K):
        import numpy as np

        q = _get_embedder().encode([query], normalize_embeddings=True,
                                   convert_to_numpy=True)
        scores, ids = self.index.search(q.astype("float32"), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0])
                if i >= 0]

    def bm25_search(self, query, k=BM25_K):
        import numpy as np

        s = self.bm25.scores(query)
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]

    @staticmethod
    def _norm(pairs):
        if not pairs:
            return {}
        vals = [s for _, s in pairs]
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {i: 1.0 for i, _ in pairs}
        return {i: (s - lo) / (hi - lo) for i, s in pairs}

    def hybrid_search(self, query, fused_k=FUSED_K):
        """Fuse dense + BM25 (50/50 after min-max normalisation)."""
        dense = self._norm(self.dense_search(query))
        bm25 = self._norm(self.bm25_search(query))
        fused = {}
        for i in set(dense) | set(bm25):
            fused[i] = 0.5 * dense.get(i, 0.0) + 0.5 * bm25.get(i, 0.0)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:fused_k]
        return [(i, fused[i], dense.get(i, 0.0), bm25.get(i, 0.0))
                for i, _ in ranked]

    def rerank(self, query, candidates, top_k=RERANK_K):
        """Cross-encoder rerank of fused candidates -> top_k chunk ids."""
        if not candidates:
            return []
        pairs = [[query, self.chunks[i]["text"]] for i, *_ in candidates]
        scores = _get_reranker().predict(pairs)
        order = sorted(range(len(candidates)), key=lambda j: -scores[j])
        return [(candidates[j][0], float(scores[j])) for j in order[:top_k]]


# ---------------------------------------------------------------- answering

def verify_citations(citations, chunks):
    """Every citation must resolve to a stored chunk id. Returns bool."""
    valid = {c["id"] for c in chunks}
    return all(c["id"] in valid for c in citations)


def _content_words(text):
    return {t for t in tokenize(text) if len(t) > 2}


def generative_answer(question, passages, max_chars=600):
    """flan-t5-small over the retrieved passages, with a quality gate.

    Returns (text, passed_gate). The gate is deliberately strict, because
    a pure word-overlap check alone passes garbage: in testing, flan-t5-small
    echoed doc fragments (a signature line, a type list, a code snippet)
    that shared words with the passages but answered nothing. So a
    generation must ALSO: end with sentence punctuation, contain no code
    prompt artefacts (">>>"), and not be a bare type/signature fragment.
    Anything failing the gate is rejected and the caller falls back to
    extractive. Default mode stays extractive.
    """
    context = "\n".join(p["text"][:800] for p in passages)
    prompt = ("Answer the question using only the context below. "
              "If the context does not answer it, say so.\n\n"
              f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
    gen = _get_generator()
    tok, mdl = gen
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=1024)
    out_ids = mdl.generate(**inputs, max_new_tokens=160)
    out = tok.decode(out_ids[0], skip_special_tokens=True).strip()
    if not (20 <= len(out) <= max_chars):
        return out, False
    if ">>>" in out or out.startswith("list["):
        return out, False
    if not re.search(r"[.!?]\s*$", out):
        return out, False
    passage_words = set()
    for p in passages:
        passage_words |= _content_words(p["text"])
    overlap = len(_content_words(out) & passage_words) / max(
        1, len(_content_words(out)))
    return out, overlap >= 0.25


_FOLLOWUP_START = ("and ", "also ", "then ", "what about", "how about",
                   "why", "what if", "so ")
_FOLLOWUP_WORDS = (" it ", " that ", " those ", " them ", " this ",
                   " its ", " their ")


class ChatSession:
    """Conversation memory: keeps exchanges; expands follow-up queries."""

    def __init__(self):
        self.history = []  # list of (question, answer_summary)

    def expand(self, question):
        """Expand a follow-up with the previous question for retrieval.

        Only genuine follow-ups expand: questions starting with a
        follow-up starter ("and …", "what about …") or containing a
        follow-up pronoun ("it", "that", "them"). A short but complete
        standalone question (e.g. "How do I use Adam?") does NOT expand,
        so its retrieval is not diluted with old context.
        """
        q = question.strip()
        low = " " + q.lower() + " "
        is_followup = q.lower().startswith(_FOLLOWUP_START) or \
            any(w in low for w in _FOLLOWUP_WORDS)
        if is_followup and self.history:
            prev_q = self.history[-1][0]
            return f"{q} (in context: {prev_q})", True
        return q, False

    def add(self, question, summary):
        self.history.append((question, summary))
        self.history = self.history[-6:]


def answer_question(index, question, session=None, generative=False,
                    debug=False):
    """Full pipeline: decompose -> hybrid -> rerank -> cited answer."""
    retrieval_q, used_history = question, False
    if session is not None:
        retrieval_q, used_history = session.expand(question)
    sub_questions = decompose_query(retrieval_q)
    queries = list(dict.fromkeys(sub_questions + [retrieval_q]))

    pool = {}
    for sub in queries:
        for idx, fused, dense, bm25 in index.hybrid_search(sub):
            if idx not in pool or fused > pool[idx][0]:
                pool[idx] = (fused, dense, bm25)
    fused_ranked = sorted(pool.items(), key=lambda kv: -kv[1][0])[:FUSED_K]
    reranked = index.rerank(retrieval_q, [(i,) for i, _ in fused_ranked])

    citations = []
    for rank, (idx, ce_score) in enumerate(reranked, 1):
        c = index.chunks[idx]
        citations.append({
            "rank": rank, "id": idx, "doc_id": c["doc_id"],
            "title": c["title"], "section": c["section"], "url": c["url"],
            "source": c["source"], "rerank_score": round(ce_score, 4),
            "text": c["text"],
        })
    verified = verify_citations(citations, index.chunks)

    mode = "extractive"
    gen_text, gen_passed = None, None
    if generative and citations:
        gen_text, gen_passed = generative_answer(
            question, [index.chunks[c["id"]] for c in citations])
        mode = "generative" if gen_passed else "extractive (generative rejected)"

    result = {
        "question": question,
        "sub_questions": sub_questions,
        "used_history": used_history,
        "mode": mode,
        "generative_text": gen_text if gen_passed else None,
        "generative_gate_passed": gen_passed,
        "citations": citations,
        "citations_verified": verified,
    }
    if debug:
        result["debug"] = {
            "retrieval_query": retrieval_q,
            "fused_candidates": [
                {"id": i, "fused": round(v[0], 4)}
                for i, v in fused_ranked[:10]],
        }
    if session is not None:
        summary = (gen_text if gen_passed else
                   citations[0]["text"][:200] if citations else "")
        session.add(question, summary)
    return result
