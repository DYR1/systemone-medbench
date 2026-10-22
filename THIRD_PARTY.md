# Third-party attribution

Vendored files retain their upstream licenses. They are provided as **source snapshots**,
not independent models developed by the benchmark authors. No pretrained weights are bundled.
The source inventory and exact content hashes are in `vendor/SOURCE_HASHES.json`.

| Directory | Upstream | License |
|---|---|---|
| `vendor/kev` | [jaredpalmer/kev](https://github.com/jaredpalmer/kev) | Apache-2.0 |
| `vendor/semif` | [TheoLeeCJ/SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev) | MIT |
| `vendor/anyjev` | [nokia-applied-research/AnyJev](https://github.com/nokia-applied-research/AnyJev) | Apache-2.0 |
| `vendor/gliner` | [fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2) | Apache-2.0 |
| `vendor/openjev` | [razorback16/openjev](https://github.com/razorback16/openjev) | Apache-2.0 |
| `vendor/startlux_v11` | [StartLuxLabs/StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision), commit `83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01` | Apache-2.0 |

For the five legacy packages, the evaluated local source snapshot is identified by file
hashes; an unrecorded upstream commit is not invented. Only their Python packages and original
license notices are copied, without benchmark data, cache files or weights. The vendor code is
not edited by the benchmark release. Integration changes live in `scripts/native/`.

StartLux v1's separately supplied `startlux_decision` source lacked a license in the available
snapshot and is not redistributed here. Obtain it from its authors under the applicable terms.
D3 and PPLX native runtimes are supplied with their model checkpoints and are not copied here.
OneJev/qev is an external extension dependency, including the official checkpoint patch.

Checkpoint licenses and access terms are separate from code licenses. The data collection is
also separately licensed; see the HF dataset card and `docs/DATA_ACCESS.md`. A dataset card's
license declaration does not eliminate constituent-source obligations or guarantee suitability
for a clinical application. No affiliation or endorsement by model or dataset authors is implied.
