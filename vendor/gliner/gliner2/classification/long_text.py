"""Long-document classification: chunk, score per chunk, aggregate logits, then
decode ONCE on the aggregate.

The ordering is the whole correctness argument. If you decoded each chunk and
merged the *decisions*, a cross-task constraint could be satisfied within every
chunk yet violated by the union. Aggregating logits first and decoding once
guarantees the returned assignment satisfies the constraints on the aggregate
evidence.
"""
from __future__ import annotations

from ..inference.chunking import split_text_into_chunks
from ..processing.word_splitter import word_splitter_from
from .scoring import ClassificationScores

_AGGREGATIONS = ("max", "mean", "first")


def _aggregate(values, mode):
    if mode == "max":
        return max(values)
    if mode == "mean":
        return sum(values) / len(values)
    return values[0]  # "first"


def aggregate_scores(scores_list, compiled, text, mode):
    if mode not in _AGGREGATIONS:
        raise ValueError(f"aggregate must be one of {_AGGREGATIONS}")
    if not scores_list:
        raise ValueError("cannot aggregate an empty list of chunk scores")
    tasks = {}
    for spec in compiled.task_specs:
        per_label = {}
        for label in spec.label_names:
            per_label[label] = _aggregate(
                [s.tasks[spec.name][label] for s in scores_list], mode)
        tasks[spec.name] = per_label
    return ClassificationScores(
        text=text, tasks=tasks, fingerprint=compiled.fingerprint,
        specs={s.name: s for s in compiled.task_specs})


def classify_long(classifier, text, schema, *, config=None, active=None,
                  chunk_size=384, chunk_overlap=64, aggregate="max"):
    """Chunk ``text``, score each chunk in batch, aggregate per-label logits,
    then decode exactly once on the aggregate."""
    from .engine import ClassificationConfig

    config = config or ClassificationConfig()
    compiled = classifier.compile_schema(schema)
    chunks = split_text_into_chunks(
        text,
        chunk_size,
        chunk_overlap,
        word_splitter=word_splitter_from(classifier),
    )
    chunk_texts = [c.text for c in chunks]
    scores_list = classifier.batch_score(chunk_texts, compiled, config=config)
    aggregated = aggregate_scores(scores_list, compiled, text, aggregate)
    return classifier.decode(aggregated, compiled, active=active, config=config)
