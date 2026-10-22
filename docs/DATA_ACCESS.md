# Data access and redistribution

This is a **collection of derived evaluations with different source licenses**, not a single
new dataset with unrestricted rights. Files retain their upstream obligations. No permission
to redistribute restricted records is implied by permission to use them locally or in a paper.

## Public data

The public Jev-format release is available at
**[yrdu/systemone-medbench](https://huggingface.co/datasets/yrdu/systemone-medbench)**.
The initial release is pinned to commit `f704b2a1785770e5a673d9725f8fb605f2f22e01`.
See the repository [quick start](../README.md#quick-start-python-310-linux-for-gpu-inference)
for the download and validation commands.

| Dataset | Entries | Declared source license |
|---|---:|---|
| CARE-Bench | 135 | CC BY-NC 4.0; source terms remain applicable |
| Ramaswamy | 960 | CC0 1.0 |
| MedCalc | 41 | CC BY-SA 4.0 |
| MedQA | 252 | CC BY 4.0 (distributed source snapshot) |
| MedXpertQA | 130 | MIT |
| SCT | 78 | MIT (public bundled dataset) |
| MedEC | 509 | CC BY 4.0; public MS subset only |
| MedVAL | 388 | MIT; retain constituent-source terms |
| RxSafe | 915 | CC0 1.0 |
| CTG-21 | 1,000 | CC BY 4.0; original UCI 21-feature table |
| **Public total** | **4,408** | Dataset-specific, not one blanket license |

Source URLs and recorded revisions are in `configs/dataset_sources.json` and the HF dataset
card. The frozen original requests are copied byte-for-byte; no extra field is removed from
medical evidence to make publication convenient. Private operational files are not copied.
The adaptations select paper cohorts, define candidate questions, attach source-derived
references, and create six presentation/decision variants. They are not new clinical adjudications.

## Withheld from the public repository

| Dataset | Entries | Reason |
|---|---:|---|
| ESI | 1,000 | Includes 931 MIMIC-derived records; the 69 public scenarios are not substituted for the full cohort |
| MIMIC-IV-ED | 1,000 | Credentialed source records |
| ICD-proxy | 1,000 | MIMIC-derived evidence |
| MIMIC-CDM | 1,000 | Credentialed source records |
| VitalDB-MAP | 334 | Provider page lists CC BY-NC-SA but also disclosure restrictions; permission needs clarification |

MIMIC terms are documented by [PhysioNet's credentialed data license](https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/).
VitalDB's additional conditions are on its [official overview/DUA page](https://vitaldb.net/dataset/?query=overview).
This release conservatively withholds these payloads, labels, mappings and examples. The public
paper-scope metadata contains counts and file hashes, not the withheld clinical records.

Authorized researchers who already possess the exact frozen Jev-format paper artifact can
import it to a private location and verify its hashes:

```bash
python bench.py import-authorized --source /private/authorized-artifact/data/jev \
  --output /private/full-paper-jev --confirm-authorized
python bench.py validate --data /private/full-paper-jev
```

This is an **importer of an existing frozen artifact**, not a raw MIMIC reconstruction pipeline.
It neither bypasses credentialing nor downloads restricted records. A fully independent
rebuild from raw credentialed datasets is not included in this public package. Contact the
dataset providers and benchmark maintainers through their authorized access processes.

Publishing only the public subset does not reproduce the full paper's numerical conclusions.
The manifest and reports explicitly name the evaluation scope. Never upload a privately
imported full artifact, its run events or its complete examples to a public HF/GitHub repository.
