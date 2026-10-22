# Complete public examples

Each subdirectory contains `request.json` (the exact model input) and `reference.json`
(the detached source-derived answer). A reference is not a simulated model response and
does not prescribe a target probability. Do not concatenate it to the request.

| Dataset | Input | Reference |
|---|---|---|
| CARE-Bench | [request](care_bench/request.json) | [reference](care_bench/reference.json) |
| Ramaswamy | [request](ramaswamy_triage/request.json) | [reference](ramaswamy_triage/reference.json) |
| MedCalc | [request](medcalc_acute_assessment/request.json) | [reference](medcalc_acute_assessment/reference.json) |
| MedQA | [request](medqa_decision/request.json) | [reference](medqa_decision/reference.json) |
| MedXpertQA | [request](medxpertqa_acute/request.json) | [reference](medxpertqa_acute/reference.json) |
| SCT | [request](sct_acute/request.json) | [reference](sct_acute/reference.json) |
| MedEC | [request](medec/request.json) | [reference](medec/reference.json) |
| MedVAL | [request](medval_review/request.json) | [reference](medval_review/reference.json) |
| RxSafe | [request](rxsafe/request.json) | [reference](rxsafe/reference.json) |
| CTG-21 | [request](ctg_uci21/request.json) | [reference](ctg_uci21/reference.json) |

These are selected from the released paper cohorts, not newly synthesized medical examples.
The five withheld datasets have no public clinical examples here. Their absence is intentional.
Data licenses apply to these examples too; see [data access](../docs/DATA_ACCESS.md).
