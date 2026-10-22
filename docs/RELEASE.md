# Release and verification notes

The public dataset is available at
[yrdu/systemone-medbench](https://huggingface.co/datasets/yrdu/systemone-medbench), initially
published at commit `f704b2a1785770e5a673d9725f8fb605f2f22e01`. This code package remains a
local GitHub publication candidate. Publish only the corresponding release directory,
not private benchmark folders, run logs or authorized full-data imports.

## Publication checklist

1. Confirm the paper title/version and GitHub repository owner. Dataset examples already point
   to the published HF repository and pin its immutable initial release commit.
2. Confirm the original benchmark code's MIT licensing choice and retain all upstream notices.
   Data licensing is per dataset; no blanket unrestricted license is claimed.
3. Do not add the five withheld dataset payloads without explicit redistribution clearance.
   If permission is later obtained, update the source notices, release scope and hashes together.
4. Run `python bench.py validate --data ...`, the test suite, and the release scanner.
   Manually review any newly added files, especially model/API responses and examples.
5. Upload the HF directory as a dataset repository and the GitHub directory as a code repository.
   Do not upload personal cache directories, `.env`, weights or run artifacts.
6. Keep the release's hash manifest and verification notes. New prompts, source snapshots,
   model backends or scoring policies should be versioned rather than silently replacing files.

## What is verified locally

- Every released inference file, detached-reference file and mapping file matches the frozen
  source bytes and recorded SHA-256 value.
- All 29,582 request hashes and parent/variant identities are checked, including 10-option
  numeric labels, binary option completeness and shuffled semantic alignment.
- Synthetic end-to-end tests exercise API serialization/parsing, resume without duplicate
  calls, sharding, all four metric dimensions, error refusal and matched length exclusion.
- The real public data pass a complete sharding dry run and API/native launch-plan checks.
- The release is scanned for common secret formats, private keys, personal machine paths,
  internal hosts, caches and weight files. Matching secret values are never printed.

No fresh GPU/model experiment or paid API request is made during packaging. A passed offline
test is not evidence that every upstream model works on every environment. Install the correct
backend and run each model's pilot before full evaluation. Model availability is controlled by
upstream providers; historical route snapshots can become unavailable.

Strict tensor-content reproducibility requires immutable checkpoints or full weight hashes.
The runtime's lightweight size/mtime fingerprint alone is not a substitute for those hashes.
