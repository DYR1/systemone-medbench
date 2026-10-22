# Model interfaces and evaluation commands

Every adapter starts from the same ordered `state` and `questions.decision` object.
`model` is routing metadata, not evidence. No adapter receives reference labels.
Native JSON compatibility and content-preserving textual adaptation are **not the same claim**.

## Interface summary

| Family | Backbone/size | Entry and decision readout | OBS treatment |
|---|---|---|---|
| Jev 1.13 / 1.14 | Undisclosed | System One API; native Choice candidate probabilities | Native Noul affirmative probability |
| Microsoft-Decision-1 / d1-ultra / GPT-6 Luna Decisions | Provider-controlled | OpenRouter decision endpoint; verify exact response model | Native Noul when supported by route |
| Kev (4B) | Qwen3.5-4B-Base + trained pointer head | `SystemOneRequest → to_record → encode(strict=True) → probs`; unrounded head distribution | Native false/true head, affirmative index 1 |
| SemIf (4B) | Qwen3.5-4B-Base | Full question JSON + structured state + criteria passed to official `direct.score`; internal letter slots | Two categorical criteria with full descriptions, not native Noul |
| AnyJev (4B) | Qwen3-4B | Full question JSON + structured state passed to `Question.choice`, `Decider(level='raw')` | `Question.noul` receives both criteria and explicit Yes/No mapping |
| GLiNER2.5-Decide | DeBERTa-v3-large-based encoder | Full state/questions JSON as text; `ID: description` class schema; all native classification logits softmaxed | Two categorical criteria, not native Noul |
| OpenJev (26B total / 4B active) | DiffusionGemma-26B-A4B-it | Official schema, template and canvas; HF BF16 eager native-slot readout | Native typed question encoded by the same schema |
| StartLux v1 (0.8B/4B/9B/27B) | Model-specific checkpoint | Official `StartLuxDecision.decide`; saved formatter and temperatures | Native Noul |
| StartLux v1.1 Instant (0.8B/4B/9B/27B) | Model-specific checkpoint | Official `Decider` + `TransformersBackend`; BF16 bidirectional readout | Native Noul; thinking and extra NONE checking disabled |
| D3 / mini / lite / nano | Checkpoint-specific (27B for the main D3, 4B for mini) | Checkpoint `D3.prepare → run → respond`; reference path, no permutation averaging | Native Noul |
| PPLX-Decider v1.1 (27B) | Checkpoint + trained readout | `DecisionModel.predict([{state, question}]) → answer`; saved temperature, noncausal attention | Native false/true ordering |
| PPLX-Decider v1.2 (27B) | Checkpoint + trained readout | `Decider.from_pretrained(..., fast=False).decide(body)`; protocol-v3 marker readout | Native false/true ordering |
| Elephant-1 / Mammoth-1 | 9B / 27B | Official patched `qev.DecisionEngine.decide`; `debias=1`, native calibration | Native yes/no ordering |

Elephant/Mammoth adapters are provided as extensions; inclusion of a runnable adapter does not
mean their experiments were complete or included in a particular paper revision. Likewise,
the GPT-6 Luna Decisions profile was investigated but must not be described as a successful run
without its own completed audit. The package contains methods, not invented model results.

### What the textual adapters actually receive

SemIf and AnyJev receive `question=json.dumps(original_question, ensure_ascii=False)`,
the original structured `state`, and an ordered candidate list. The full question includes
`type`, `instructions` and **all criterion descriptions**. For OBS, `criteria.true` and
`criteria.false` are never replaced with bare Boolean strings. Internal answer slots may
remain letters even when visible candidate IDs change; the relabeling test still changes
the visible content. GLiNER receives the full ordered `{"state": ..., "questions": ...}`
serialization plus a class schema. Results therefore describe model–adapter combinations.

The `model` route identifier is omitted from inference text in all content adapters.
No common LLM prompt, answer prefill or generative sampling is added to a native decision API.

## Dependencies and sources

Use separate environments. Install a PyTorch build compatible with your actual hardware first;
the commands below do not authorize replacing vendor kernels. Baseline versions are provided
for replication, not as a claim that arbitrary later packages are bitwise equivalent.

- D3 and PPLX reference runtimes used **Transformers 5.17.0**. Install their own checkpoint
  dependencies (including tokenizer, processor, safetensors, accelerate, Pillow/vision support
  if imported by the runtime even for text). Keep `source/`, tokenizer and config files.
- StartLux v1.1 used **Transformers 5.8.x**; its worker checks this major/minor version.
  `vendor/startlux_v11/pyproject.toml` supplies upstream dependencies. Do not install the vLLM
  optional backend for this BF16 Transformers protocol.
- Kev/SemIf need native Qwen3.5 support; AnyJev uses Qwen3 support. Common optional imports include
  `torch`, `transformers`, `accelerate`, `safetensors`, `sentencepiece`, `pydantic`, and `peft`
  (where the model requires it). GLiNER also imports its tokenizer/schema dependencies.
- OpenJev requires a Transformers build exporting `DiffusionGemmaForBlockDiffusion` and
  `experts_implementation='eager'`. Its published benchmark path here is **HF eager**, not
  a claim of equivalence to the official vLLM server. Canvas seed is 1234. A future backend
  replacement is a new protocol and must be labeled as such.
- Elephant/Mammoth used **Transformers 5.19.0** and OneJev commit
  `81ce62f1597c91e46767d4d02d6ac2e18534fe94` plus the checkpoint's official `serve.patch`.
  Follow the checkpoint's patch instructions before passing the source directory to `--source`.
  The launcher checks `configs/qev_expected_source_hashes.json`; a different implementation
  requires an explicitly versioned adapter/protocol, not a silent source substitution.

The release is CPU-tested for data, protocol and scoring; it was not re-run on GPUs during
packaging. Historical adapters were ported from the evaluation code. Always run the supplied
pilot in the target environment before spending compute on a full run.

## Download references

Use `hf download REPO --revision IMMUTABLE_COMMIT --local-dir /path/to/checkpoint` after
accepting any upstream terms. Do not use moving `main` for an experiment you intend to compare.
For snapshots without a recorded public commit below, select the intended release explicitly
and retain complete checkpoint/source hashes. Internal distribution identifiers are not included.

| Artifact | Public repository | Historical immutable revision where recorded |
|---|---|---|
| Kev adapter | `jaredpalmer/kev-4b` | `139fdd94f1b6a6ad80cc15e08fcb99cac885a101` |
| Kev/SemIf base | `Qwen/Qwen3.5-4B-Base` | `1001bb4d826a52d1f399e183466143f4da7b741b` |
| AnyJev base | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| GLiNER | `fastino/GLiNER2.5-Decide` | `5a7adf72a23b4d311abae6ce050d7f0012bb3416` |
| OpenJev base | `google/diffusiongemma-26B-A4B-it` | `f7f5b7f5fa82ffc52addd066915886d497f5517b` |
| StartLux | `startlux-models/StartLux-Decision-{SIZE}` / `StartLux-Decision-v1.1-{SIZE}-Instant` | Select exact model release; AutoThink is not this protocol |
| D3 | `vllm-sr/d3`, `d3-mini`, `d3-lite`, `d3-nano` | Select exact release and native code |
| PPLX | `perplexity-ai/pplx-decider-v1.1-27b`, `perplexity-ai/pplx-decider-v1.2` | Keep bundled `source/src` |
| Elephant/Mammoth | `baselabs/elephant-1`, `baselabs/mammoth-1` | Keep official `serve.patch` |

Licensed upstream source snapshots for Kev, SemIf, AnyJev, GLiNER, OpenJev and StartLux v1.1
are included under `vendor/`; exact file hashes are in `vendor/SOURCE_HASHES.json`.
See [THIRD_PARTY.md](../THIRD_PARTY.md) for attribution. No training data or weights are vendored.

The local StartLux **v1** source snapshot did not include a separate license. Its adapter is
included but that upstream module is **not redistributed**. Obtain the official
`startlux_decision` module under its applicable terms, then pass its parent directory via
`--source`. `configs/startlux_v1_expected_source_hashes.json` identifies the historical code.
This is a disclosed external dependency, not an included implementation.

## Adapter-specific examples

For each model, first `prepare` a fresh work directory with the desired shard count.
Run the command once with `--devices 0 --pilot --execute`, then with the full device list
and `--execute`. Omitting `--execute` only displays the plan.

```bash
# Original StartLux; external licensed source required
python bench.py native --adapter startlux-v1 --model /path/to/startlux-4b \
  --source /path/to/official-v1-source --response-model StartLux-Decision-4B --work runs/startlux --devices 0 --pilot

# StartLux v1.1 Instant, official source included
python bench.py native --adapter startlux-v11 --model /path/to/startlux-v11-4b \
  --response-model StartLux-Decision-v1.1-4B-Instant --work runs/startlux-v11 --devices 0 --pilot

# Kev needs the separate trained head/adapter and its full base
python bench.py native --adapter kev --model /path/to/kev-4b --base-model /path/to/qwen35-4b-base \
  --response-model Kev-4B --work runs/kev --devices 0 --pilot

# SemIf: no separately trained checkpoint in this configuration
python bench.py native --adapter semif --model /path/to/qwen35-4b-base \
  --response-model SemIf-4B --work runs/semif --devices 0 --pilot
python bench.py native --adapter anyjev --model /path/to/qwen3-4b \
  --response-model AnyJev-4B --work runs/anyjev --devices 0 --pilot
python bench.py native --adapter gliner --model /path/to/gliner \
  --response-model GLiNER2.5-Decide --work runs/gliner --devices 0 --pilot
python bench.py native --adapter openjev --model /path/to/diffusiongemma \
  --response-model OpenJev --work runs/openjev --devices 0 --pilot

# PPLX v1.1 and v1.2 are distinct APIs, not aliases
python bench.py native --adapter pplx-v11 --model /path/to/pplx-v11 \
  --response-model pplx-decider-v1.1-27b --work runs/pplx-v11 --devices 0 --pilot
python bench.py native --adapter pplx-v12 --model /path/to/pplx-v12 \
  --response-model pplx-decider-v1.2 --work runs/pplx-v12 --devices 0 --pilot

# Extension: officially patched OneJev source, not a modified generic LLM evaluator
python bench.py native --adapter qev --model /path/to/elephant-1 --source /path/to/patched-OneJev \
  --response-model elephant-1-9B --work runs/elephant --devices 0 --pilot
```

### Critical controls

- StartLux v1.1 uses `think_mode=never`, `none_check=false`, `allow_none=false` and
  `return_details=true`. These disable an **extra internal checking mechanism**; they do not
  remove the benchmark's explicit NONE/ABSTAIN candidates. AutoThink is out of scope.
- D3 uses `D3_FAST=0`, native saved temperatures, batch size 1 and no permutation averaging.
- PPLX v1.1 uses its native 8,192-token cap. v1.2 uses its native 262,144-token cap. No input is
  truncated to fit. Runtime/hardware OOM is distinct from a model context-length error.
- qev uses `debias=1`, native rounding/calibration, 16,384 branch-token and 65,536 request-token
  limits, and the official checkpoint patch. The release does not increase these limits.
- Original legacy adapters retain the evaluated 32,768-token guard. SemIf/GLiNER binary
  adapters are not renamed “native Noul”.
- All probabilities are recorded before benchmark OBS normalization. Selected-answer
  probability—not a separately returned `confidence` field—is used for retention analysis.

The launchers fingerprint code, metadata, model file sizes/mtimes and configuration. This
detects ordinary changes but is **not a cryptographic digest of all tensor contents**.
Archive full weight checksums or immutable HF revisions for strict cross-machine replication.
