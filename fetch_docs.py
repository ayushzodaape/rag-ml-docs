"""Fetch the ML-docs corpus: official scikit-learn and PyTorch documentation
pages only (trusted sources). Any page that cannot be fetched from the
official site is skipped and logged — never substituted with another source.

Usage:  ~/workspace/ml_env/bin/python fetch_docs.py
Output: corpus/docs.json (sections per page) + corpus/fetch_log.json
"""

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
CORPUS_DIR = BASE / "corpus"
DOCS_PATH = CORPUS_DIR / "docs.json"
LOG_PATH = CORPUS_DIR / "fetch_log.json"

UA = "rag-ml-docs local demo (educational, non-commercial)"
DELAY = 0.4  # polite pause between requests

# (doc_id, title, url) — official docs only.
SKLEARN = "https://scikit-learn.org/stable/modules/generated/"
PYTORCH = "https://docs.pytorch.org/docs/stable/"

PAGES = [
    # ---- scikit-learn: estimators ----
    ("sklearn-randomforest", "sklearn.ensemble.RandomForestClassifier",
     SKLEARN + "sklearn.ensemble.RandomForestClassifier.html"),
    ("sklearn-histgb", "sklearn.ensemble.HistGradientBoostingClassifier",
     SKLEARN + "sklearn.ensemble.HistGradientBoostingClassifier.html"),
    ("sklearn-logreg", "sklearn.linear_model.LogisticRegression",
     SKLEARN + "sklearn.linear_model.LogisticRegression.html"),
    ("sklearn-linreg", "sklearn.linear_model.LinearRegression",
     SKLEARN + "sklearn.linear_model.LinearRegression.html"),
    ("sklearn-svc", "sklearn.svm.SVC",
     SKLEARN + "sklearn.svm.SVC.html"),
    ("sklearn-knn", "sklearn.neighbors.KNeighborsClassifier",
     SKLEARN + "sklearn.neighbors.KNeighborsClassifier.html"),
    ("sklearn-kmeans", "sklearn.cluster.KMeans",
     SKLEARN + "sklearn.cluster.KMeans.html"),
    ("sklearn-pca", "sklearn.decomposition.PCA",
     SKLEARN + "sklearn.decomposition.PCA.html"),
    ("sklearn-tree", "sklearn.tree.DecisionTreeClassifier",
     SKLEARN + "sklearn.tree.DecisionTreeClassifier.html"),
    ("sklearn-gnb", "sklearn.naive_bayes.GaussianNB",
     SKLEARN + "sklearn.naive_bayes.GaussianNB.html"),
    # ---- scikit-learn: preprocessing / pipelines ----
    ("sklearn-scaler", "sklearn.preprocessing.StandardScaler",
     SKLEARN + "sklearn.preprocessing.StandardScaler.html"),
    ("sklearn-onehot", "sklearn.preprocessing.OneHotEncoder",
     SKLEARN + "sklearn.preprocessing.OneHotEncoder.html"),
    ("sklearn-pipeline", "sklearn.pipeline.Pipeline",
     SKLEARN + "sklearn.pipeline.Pipeline.html"),
    ("sklearn-columntransformer", "sklearn.compose.ColumnTransformer",
     SKLEARN + "sklearn.compose.ColumnTransformer.html"),
    ("sklearn-imputer", "sklearn.impute.SimpleImputer",
     SKLEARN + "sklearn.impute.SimpleImputer.html"),
    ("sklearn-selectkbest", "sklearn.feature_selection.SelectKBest",
     SKLEARN + "sklearn.feature_selection.SelectKBest.html"),
    # ---- scikit-learn: model selection / metrics ----
    ("sklearn-train-test-split", "sklearn.model_selection.train_test_split",
     SKLEARN + "sklearn.model_selection.train_test_split.html"),
    ("sklearn-gridsearch", "sklearn.model_selection.GridSearchCV",
     SKLEARN + "sklearn.model_selection.GridSearchCV.html"),
    ("sklearn-cross-val", "sklearn.model_selection.cross_val_score",
     SKLEARN + "sklearn.model_selection.cross_val_score.html"),
    ("sklearn-accuracy", "sklearn.metrics.accuracy_score",
     SKLEARN + "sklearn.metrics.accuracy_score.html"),
    ("sklearn-classification-report", "sklearn.metrics.classification_report",
     SKLEARN + "sklearn.metrics.classification_report.html"),
    ("sklearn-confusion-matrix", "sklearn.metrics.confusion_matrix",
     SKLEARN + "sklearn.metrics.confusion_matrix.html"),
    ("sklearn-roc-auc", "sklearn.metrics.roc_auc_score",
     SKLEARN + "sklearn.metrics.roc_auc_score.html"),
    # ---- PyTorch: nn module ----
    ("torch-linear", "torch.nn.Linear",
     PYTORCH + "generated/torch.nn.Linear.html"),
    ("torch-conv2d", "torch.nn.Conv2d",
     PYTORCH + "generated/torch.nn.Conv2d.html"),
    ("torch-module", "torch.nn.Module",
     PYTORCH + "generated/torch.nn.Module.html"),
    ("torch-sequential", "torch.nn.Sequential",
     PYTORCH + "generated/torch.nn.Sequential.html"),
    ("torch-relu", "torch.nn.ReLU",
     PYTORCH + "generated/torch.nn.ReLU.html"),
    ("torch-dropout", "torch.nn.Dropout",
     PYTORCH + "generated/torch.nn.Dropout.html"),
    ("torch-batchnorm1d", "torch.nn.BatchNorm1d",
     PYTORCH + "generated/torch.nn.BatchNorm1d.html"),
    ("torch-softmax", "torch.nn.Softmax",
     PYTORCH + "generated/torch.nn.Softmax.html"),
    # ---- PyTorch: losses / optimizers ----
    ("torch-crossentropy", "torch.nn.CrossEntropyLoss",
     PYTORCH + "generated/torch.nn.CrossEntropyLoss.html"),
    ("torch-mse", "torch.nn.MSELoss",
     PYTORCH + "generated/torch.nn.MSELoss.html"),
    ("torch-adam", "torch.optim.Adam",
     PYTORCH + "generated/torch.optim.Adam.html"),
    ("torch-sgd", "torch.optim.SGD",
     PYTORCH + "generated/torch.optim.SGD.html"),
    ("torch-steplr", "torch.optim.lr_scheduler.StepLR",
     PYTORCH + "generated/torch.optim.lr_scheduler.StepLR.html"),
    # ---- PyTorch: data / autograd / utils ----
    ("torch-data", "torch.utils.data (Dataset & DataLoader)",
     PYTORCH + "data.html"),
    ("torch-tensor", "torch.tensor",
     PYTORCH + "generated/torch.tensor.html"),
    ("torch-autograd", "Automatic Differentiation (autograd)",
     PYTORCH + "notes/autograd.html"),
    ("torch-save", "torch.save",
     PYTORCH + "generated/torch.save.html"),
    ("torch-load", "torch.load",
     PYTORCH + "generated/torch.load.html"),
    ("torch-manual-seed", "torch.manual_seed",
     PYTORCH + "generated/torch.manual_seed.html"),
    ("torch-no-grad", "torch.no_grad",
     PYTORCH + "generated/torch.no_grad.html"),
]

MAX_SECTION_CHARS = 60000
MAX_CODE_CHARS = 3000


def _redirect_target(html, base_url):
    """Return the redirect target of a meta-refresh/canonical stub, or None."""
    import html as html_mod
    from urllib.parse import urljoin

    m = re.search(
        r'<meta[^>]+http-equiv=["\']refresh["\'][^>]+url=([^"\'>]+)',
        html, re.IGNORECASE)
    if m:
        return urljoin(base_url, html_mod.unescape(m.group(1).strip()))
    m = re.search(r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)',
                   html, re.IGNORECASE)
    if m:
        target = urljoin(base_url, html_mod.unescape(m.group(1)))
        if target != base_url:
            return target
    return None


def fetch_html(url):
    """Fetch a page, following meta-refresh/JS-redirect stubs (PyTorch
    serves /stable/ as a redirect stub to a versioned /2.x/ page)."""
    current = url
    for _ in range(3):
        req = urllib.request.Request(current, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    return None, f"HTTP {resp.status}", current
                html = resp.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            return None, f"HTTP {e.code}", current
        except Exception as e:  # noqa: BLE001 - network fetch, report
            return None, f"{type(e).__name__}: {e}", current
        target = None
        # Only a short stub page is a redirect: full content pages also
        # carry a canonical link (pointing back at /stable/), which must
        # NOT be followed, or stable <-> versioned ping-pong forever.
        if len(html) < 5000:
            target = _redirect_target(html, current)
        if target and target != current:
            current = target
            time.sleep(0.2)
            continue
        return html, None, current
    return None, "redirect loop", current


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def parse_sections(html):
    """Split a sphinx docs page into (section_heading, text) pairs.

    Walks headings/paragraphs/lists/code in document order inside the main
    content container, grouping everything under its nearest h2/h3 heading.
    Code examples are kept inline as fenced blocks.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("div", attrs={"role": "main"}) or soup.find("article") \
        or soup.body
    if main is None:
        return []
    for tag in main.find_all(["script", "style", "nav", "footer", "form"]):
        tag.decompose()

    sections = []
    current_heading = "Overview"
    current_parts = []

    def flush():
        nonlocal current_parts
        text = "\n".join(p for p in current_parts if p).strip()
        if len(text) > 40:
            sections.append((current_heading, text[:MAX_SECTION_CHARS]))
        current_parts = []

    for el in main.find_all(["h1", "h2", "h3", "p", "li", "pre", "dt", "dd"]):
        if el.name in ("h2", "h3"):
            flush()
            current_heading = clean_text(el.get_text()) or current_heading
        elif el.name == "h1":
            continue  # page title, recorded separately
        elif el.name == "pre":
            code = clean_text(el.get_text())
            if code:
                current_parts.append("```\n" + code[:MAX_CODE_CHARS] + "\n```")
        else:
            text = clean_text(el.get_text())
            # skip pure navigation junk
            if text and text not in ("¶", "Permalink to this heading"):
                current_parts.append(text)
    flush()
    return sections


def main():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    docs, log = [], {"attempted": len(PAGES), "fetched": [], "skipped": []}
    for doc_id, title, url in PAGES:
        source = "scikit-learn" if "scikit-learn.org" in url else "pytorch"
        html, err, resolved = fetch_html(url)
        if html is None:
            log["skipped"].append({"doc_id": doc_id, "url": url, "reason": err})
            print(f"SKIP {doc_id}: {err}")
        else:
            sections = parse_sections(html)
            total_chars = sum(len(t) for _, t in sections)
            # A generated API page legitimately has a single section (the
            # whole docstring under "Overview"); total chars is the real
            # stub detector, not the section count.
            if total_chars < 500:
                log["skipped"].append(
                    {"doc_id": doc_id, "url": url,
                     "reason": f"too little content ({len(sections)} sections, "
                               f"{total_chars} chars)"})
                print(f"SKIP {doc_id}: too little content")
            else:
                docs.append({"doc_id": doc_id, "title": title, "url": url,
                             "resolved_url": resolved,
                             "source": source,
                             "sections": [{"heading": h, "text": t}
                                          for h, t in sections]})
                log["fetched"].append(doc_id)
                print(f"OK   {doc_id}: {len(sections)} sections, "
                      f"{total_chars} chars")
        time.sleep(DELAY)

    DOCS_PATH.write_text(json.dumps(docs, indent=1), encoding="utf-8")
    LOG_PATH.write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"\nFetched {len(docs)}/{len(PAGES)} pages -> {DOCS_PATH}")
    print(f"Skipped: {[s['doc_id'] for s in log['skipped']]}")


if __name__ == "__main__":
    main()
