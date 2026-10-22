# Evaluation protocol

## Scope and data structure

The full paper scope is 15 datasets, 8,742 dataset-qualified entries and 65,252 requests.
The public release has 10 datasets, 4,408 entries and 29,582 requests. The manifest defines
the actual denominator. “Entry” is not synonymous with an independent patient; some tasks
reuse evidence under demographic or candidate variations. Nothing in this release expands
the experiment to images, agents, generated reasoning, score questions or sepsis.

```text
data/jev/<category>/<dataset>/
  answers.jsonl                       Canonical parent references
  choice_letters/jev_requests.jsonl    Inference records
  choice_letters/answers.jsonl         Detached variant references
  choice_letters/mappings.jsonl        Display/canonical identities
  choice_numbers/...
  choice_shuffled/...
  binary_ovr/...
  choice_none/...                     Eligible B datasets only
  choice_abstain/...                  B datasets only
```

An inference row contains `id`, `parent_id`, `body` and `request_sha256`.
`body` has `model`, `state`, and `questions.decision`. `state` is the original structured
evidence object; its field descriptions are included where present. `questions.decision`
contains `type`, `instructions`, and ordered `criteria`. For Choice these are display IDs
and candidate meanings. For Noul these are the full `true` and `false` propositions.
IDs join the three files but are not sent to a native model as evidence.

The hash is SHA-256 of UTF-8 `json.dumps(body, ensure_ascii=False, allow_nan=False)` using
Python's default separators and **preserving dictionary order**. Sorting candidate keys can
change the evaluated presentation and is forbidden. Route metadata can change to identify
the requested model, but original and wire-request hashes are both logged.

References may contain `label`, `acceptable_labels`, or SCT `option_scores`. Accuracy is
membership in the accepted set; for SCT it is membership among the highest expert-score
options, **not partial-credit accuracy**. Macro-F1 uses uniquely accepted references and a
fixed canonical class set, with zero F1 for an unsupported class. The scorer reports the
eligible F1 count. Lower-level historical partial-credit helpers are not used as the reported
paper accuracy.

## Six input variants

| Variant | Intervention | Identity/reference rule |
|---|---|---|
| `choice_letters` | Original A/B/C/... | Canonical reference |
| `choice_numbers` | Only display labels become 1/2/3/... | Map numeric IDs back; ten candidates remain 1–10 |
| `choice_shuffled` | Fixed seeded permutation of candidates with A/B/... display IDs | Map displayed positions back to candidate meanings |
| `binary_ovr` | Independently judge each original candidate using full true/false criteria | Associate every affirmative probability with its canonical candidate |
| `choice_none` | Replace the unique correct candidate's text with NONE | Correct modified answer is its retained slot; no semantic-preservation claim |
| `choice_abstain` | Append ABSTAIN while retaining every original candidate | ABSTAIN is withholding, not a new correct diagnosis |

Correct-answer replacement covers ICD-proxy, MedQA, MedXpertQA and MIMIC-CDM in the full
paper (2,382 entries). Items need more than three candidates and a unique accepted reference.
Optional abstention covers all six B datasets (2,501 entries). In the public release these
become **382 entries across two datasets** and **501 entries across four datasets**.
Do not present the public totals as the original complete Clinical Judgment experiment.

## 1. Overall performance

Compute Accuracy and macro-F1 within each dataset, then equally average dataset-level
values for a task family and for the overall result. The overall result weights the 15
datasets equally, not the four task families equally and not entries proportionally.
The same rule applies within the explicitly named public subset.

## 2. Format robustness

For each matched entry, map variant display labels to original semantic identities first.
Decision reversal is the fraction whose selected canonical option changes.
Report the accuracy change in percentage points for numeric, shuffled and OBS formats.
The mean reversal is the mean of the three dataset-mean reversal rates.
Within-dataset accuracy variance is the population variance of the four percentage accuracies
(original, numeric, shuffled, OBS); its unit is pp², then datasets are equally averaged.

For canonical probability vectors p and p′, total variation is
`TV = 0.5 × sum_k |p_k − p′_k|`. Mean absolute probability difference and Jensen–Shannon
divergence in bits are also saved. These require the same canonical candidate set and paired
entries; NONE/ABSTAIN distributions are not silently compared as if their meanings were unchanged.

## 3. Abstention awareness

For NONE, report matched original accuracy, modified accuracy, their pp difference,
slot reversal, correctness reversal, and conditional NONE accuracy among originally correct
entries. Slot reversal changes the selected position. Correctness reversal includes **both**
correct→incorrect and incorrect→correct transitions relative to each version's reference.
It is not equivalent to decision reversal because a candidate's meaning has changed.

For ABSTAIN, let S_d be retained entries of dataset d. Report dataset-mean abstention rate,
retained count and accuracy, original full-cohort accuracy, original accuracy on exactly S_d,
and both pp differences. The first delta reflects selection plus answer changes; the same-subset
delta isolates net answer changes. A dataset with zero retained entries has undefined retained
accuracy and is omitted from retained-accuracy means, not assigned zero. Coverage is still
computed for it. Pooled counts and equal-dataset rates can differ; both are explicitly labeled.

## 4. Probability-based reliability

Choice uses the probability assigned to the model's selected answer, not an API field named
`confidence`. OBS reads native Noul affirmative probabilities q_k and forms
`p_k = q_k / sum_j q_j`; its decision is `argmax_k p_k`. A native classifier may internally
use a softmax; this does not change the benchmark's **linear** OBS normalization.
An incomplete option set or an all-zero affirmative vector is invalid.

Pool all parent entries in the declared scope for this dimension. Retain entries at
probability ≥ 0.70, 0.75, 0.80, 0.85 and 0.90. Also rank globally and retain the top
5%, 10%, 15%, 20% and 25%, using floor(N × percentage / 100).
Exact ties use a fixed hash-based ID ordering, independent of reference labels. This rule
is fixed in code and must not be tuned to favor either interface. Report Accuracy/N/coverage,
per-dataset composition, within-dataset retained fractions, and probability distributions.
Do not compute macro-F1 on selective subsets or perform within-dataset ranking as a substitute
for the global matched-coverage experiment. These are selective accuracy analyses, not proof
of calibration, downstream routing benefit or medical safety.

## Errors and reproducibility

The normal report requires every planned request and every OBS candidate. No silent input
truncation, missing-as-incorrect conversion or probability imputation is allowed. The audit
identifies missing/failed requests before scoring. Resume successful outputs only if input,
code, checkpoint and protocol fingerprints agree. Preserve error attempts for diagnosis.

An explicitly declared length-error exclusion can be used with `--exclude-length-errors`:
all requests must have been attempted; only recorded length errors are eligible; affected
parent entries are removed from every format within their dataset. The resulting matched
cohort and new denominators are recorded and are **not the full-paper cohort**. Other model,
network, OOM, unattempted or invalid-probability errors cannot be excluded by this flag.

All API version identifiers and native sources are recorded. Immutable model revisions and
full weight checksums should be archived for strict replication. Independent GPU replicas
partition by parent entry, keeping all versions/OBS questions for an entry on one replica.
The evaluation does not claim bitwise identity across different hardware kernels.
