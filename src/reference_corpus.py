"""
Shared reference-corpus loader.

Used to "ground" the classifier's plagiarism verdict: a "Plagiarized"
call only means something if there's an actual source document behind
it. Without this, `HuggingFaceDetector.predict()` is just guessing
based on learned sentence style, which is what causes confident but
wrong calls on original, casual/first-person text (see
`predict_grounded()` in huggingface_detector.py for the full guard
logic).

Reference documents live in data/reference_corpus.txt, one document per
line (blank lines ignored). Drop real source material in there - course
notes, textbook excerpts, articles you actually want to check submitted
text against - and it's picked up automatically, no code changes needed.

If that file doesn't exist yet, a tiny built-in fallback corpus is used
instead so the app still runs, but grounding against only 2 sentences is
too small to be meaningful - treat the fallback as a placeholder, not a
real reference set.
"""

import os
from typing import List

DEFAULT_CORPUS_PATH = os.path.join("data", "reference_corpus.txt")

_FALLBACK_CORPUS = [
    "Artificial intelligence is the simulation of human intelligence in machines",
    "Machine learning is a subset of artificial intelligence"
]


def load_reference_corpus(path: str = DEFAULT_CORPUS_PATH) -> List[str]:
    """Load reference documents from `path`, one per line.

    Falls back to a tiny built-in placeholder corpus if the file is
    missing or empty, so the app never crashes for lack of a corpus -
    it just grounds weakly until real reference documents are added.
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if lines:
            return lines
        print(f"⚠️ {path} exists but is empty, using fallback reference corpus")
    else:
        print(f"⚠️ No reference corpus found at {path}, using fallback (add real source "
              f"documents there for meaningful plagiarism grounding)")

    return list(_FALLBACK_CORPUS)