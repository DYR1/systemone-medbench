import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi

from kev.composition import DEV_SHAPES, HELD_OUT_KEYS, TEST_SHAPES, TRAIN_SHAPES, canonical, check_group, generate as compose, sample_trees
from kev.contrastive import ORDINAL_FAMILIES, generate
from kev.data import materialize
from kev.model import fits, load_tokenizer
from kev.suite import SPLITS, digest, load_split, read_manifest, semantic_hash, validate_training, write_json, write_jsonl

BASES = ("Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-4B-Base")
FAMILIES = ("return_window", "spend_threshold", "age_eligibility", "quantity_limit")


def grouped_split(records, calibration_groups):
    groups = defaultdict(dict)
    for r in records:
        m = r["_meta"]
        groups[m["family"]].setdefault(m["group_id"], []).append(r)
    train, calibration = [], []
    for family in sorted(groups):
        units = list(groups[family].values())
        if len(units) <= calibration_groups:
            raise ValueError("not enough groups per family")
        calibration.extend(r for group in units[:calibration_groups] for r in group)
        train.extend(r for group in units[calibration_groups:] for r in group)
    return train, calibration


def legacy(pairs, seed, families=FAMILIES, source="legacy_policy", excluded=()):
    candidates, _ = generate(3 * pairs, seed, families)
    records, seen, counts = [], set(excluded), Counter()
    for a, b in zip(candidates[::2], candidates[1::2]):
        family = a["_meta"]["family"]
        hashes = {semantic_hash(a), semantic_hash(b)}
        if counts[family] >= pairs or hashes & seen:
            continue
        for r in (a, b):
            m = r["_meta"]
            m.update(source=source, group_id=m["pair_id"], variant="clean")
            m["text_sha256"] = semantic_hash(r)
        records.extend((a, b)); seen.update(hashes); counts[family] += 1
    if any(counts[f] != pairs for f in families):
        raise ValueError("insufficient unique legacy pairs")
    return records


def freeze(out, source="evals/decision-v2", transfer="evals/transfer-v2", public_train=None, synthetic_scale=1, inherit_eval=None,
           random_structures=0, groups_per_structure=8, train_styles=(0, 1), matched_arms=True, legacy_families=FAMILIES, structures_seed=None):
    """random_structures > 0: the compositional arm is generated from that many random rule trees (negation anywhere,
    held-out and locked structures excluded by canonical key) instead of the eight fixed TRAIN_SHAPES."""
    """inherit_eval: a frozen decision suite whose development/test bytes are reused verbatim (only train/calibration are
    regenerated), so results stay comparable across suite versions that differ in training data only."""
    """public_train: optional larger public training pool (a frozen suite dir). Its train+calibration partitions replace
    the source suite's public train/calibration; development/test still come from `source` so results stay comparable."""
    out, source, transfer = Path(out), Path(source), Path(transfer)
    if out.exists():
        raise FileExistsError("v3 destination already exists; choose a new version")
    original = read_manifest(source)
    old_transfer = read_manifest(transfer)
    revisions = {base: HfApi().model_info(base).sha for base in BASES}
    tokenizers = [load_tokenizer(base, revision=sha) for base, sha in revisions.items()]
    parts = {s: [] for s in SPLITS}
    for split in ("train", "calibration", "development"):
        parts[split] = [r for r in load_split(source, split) if r["_meta"]["source"] != "contrastive"]
    if public_train:
        pool = Path(public_train)
        dev_test_states = {r["_meta"]["text_sha256"] for split in ("development", "test") for r in load_split(source, split, allow_test=True)}
        for split in ("train", "calibration"):
            fresh = [r for r in load_split(pool, split) if r["_meta"]["source"] != "contrastive"]
            leaked = [r for r in fresh if r["_meta"]["text_sha256"] in dev_test_states]
            if leaked: raise ValueError(f"{len(leaked)} public {split} records collide with development/test states")
            parts[split] = fresh
        source_dirs = [pool]
    else:
        source_dirs = []
    reserved = set()
    for split in ("train", "calibration", "development"):
        reserved.update(semantic_hash(r) for r in parts[split])
    sources = [p for p in source.glob("*.json*")] + [p for p in transfer.glob("*.json*")] + [p for d in source_dirs for p in d.glob("*.json*")]
    parent_hashes = {str(p): digest(p) for p in sources}

    def admit_groups(records):
        groups = defaultdict(list)
        for r in records:
            groups[r["_meta"]["group_id"]].append(r)
        for group in groups.values():
            if group[0]["_meta"]["source"] in ("compositional", "composition_holdout"):
                check_group(group)
            hashes = {semantic_hash(r) for r in group}
            if hashes & reserved:
                raise ValueError("semantic state collision across groups; choose a new generation seed")
            for r in group:
                if not fits(materialize(r), *tokenizers):
                    raise ValueError("synthetic record exceeds the training context")
            reserved.update(hashes)
        return records

    # development synthetic records are generated first, against public states only, so they do not depend on the size
    # of the synthetic training arms (dev/test bytes stay identical across suite versions); training groups that would
    # collide with a development state are rejected by admit_groups
    if inherit_eval:
        inherited_dev = load_split(inherit_eval, "development")
        parts["development"] = inherited_dev
        reserved.update(semantic_hash(r) for r in inherited_dev)
        reserved.update(semantic_hash(r) for r in load_split(inherit_eval, "test", allow_test=True))
        dev_synthetic = []
    else:
        dev_synthetic = admit_groups(legacy(12, "v3-control-dev", excluded=reserved)) + admit_groups(compose(4, "v3-composition-dev"))
    old_train, old_cal = grouped_split(admit_groups(legacy(64 * synthetic_scale, "v3-control", families=legacy_families, excluded=reserved)), 8 * synthetic_scale)
    if random_structures:
        trees = {f"rand:{canonical(t)}": t for t in sample_trees(random_structures, f"{structures_seed or out.name}-structures")}
        new_train, new_cal = grouped_split(admit_groups(compose(groups_per_structure, "v7-composition", styles=train_styles, trees=trees)), 1)
    else:
        new_train, new_cal = grouped_split(admit_groups(compose(16 * synthetic_scale, "v3-composition")), 2 * synthetic_scale)
    if matched_arms and (len(old_train) != len(new_train) or len(old_cal) != len(new_cal)):
        raise ValueError("synthetic arms have unequal record budgets")
    parts["train"] += old_train + new_train
    parts["calibration"] += old_cal + new_cal
    parts["development"] += dev_synthetic
    transfer_dev = [r for r in load_split(transfer, "development") if r["_meta"]["source"] != "contrastive"]
    transfer_dev += admit_groups(legacy(20, "v3-legacy-transfer", ("authorization", "deadline"), source="legacy_holdout"))
    transfer_dev += admit_groups(compose(8, "v3-composition-transfer", DEV_SHAPES, styles=(1,), source="composition_holdout"))
    final_extra = admit_groups(compose(8, "v3-locked", TEST_SHAPES, styles=(2,), source="composition_holdout"))
    for split in ("train", "calibration"):
        random.Random(f"v3-{split}").shuffle(parts[split])
    manifest = {"version": 3, "base_revisions": revisions, "dataset_revisions": original["dataset_revisions"],
        "parent_files": parent_hashes, "holdout_sources": [],
        "trainable_sources": sorted({s for s in original["trainable_sources"] if s != "contrastive"}
                                    | ({s for s in read_manifest(public_train)["trainable_sources"]} if public_train else set())) + ["legacy_policy", "compositional"],
        "eval_only_sources": old_transfer["eval_only_sources"] + ["legacy_holdout", "composition_holdout"],
        "context": original["context"],
        "protocol": {"train_shapes": TRAIN_SHAPES, "transfer_shapes": DEV_SHAPES, "locked_shapes": TEST_SHAPES,
                     "train_render_styles": list(train_styles), "locked_render_styles": [2],
                     "random_structures": random_structures, "groups_per_structure": groups_per_structure if random_structures else None,
                     "structures_seed": f"{structures_seed or out.name}-structures" if random_structures else None,
                     "excluded_structure_keys": sorted(HELD_OUT_KEYS), "legacy_families_trainable": list(legacy_families),
                     "public_train_records": len(parts["train"]) - len(old_train) - len(new_train),
                     "public_train_pool": str(public_train) if public_train else str(source), "synthetic_scale": synthetic_scale,
                     "inherited_eval": str(inherit_eval) if inherit_eval else None,
                     "synthetic_records_per_arm": len(old_train) if matched_arms else {"legacy_policy": len(old_train), "compositional": len(new_train)},
                     "calibration": "shared; stratified by family and group",
                     "arm_selection": "train_sources selects public sources plus exactly one synthetic arm",
                     "primary": "macro development NLL; transfer and confident-error checks required; no automatic release",
                     "legacy_test": "Inherited v2 locked test bytes retained without inspecting examples; only new structures added to transfer test",
                     "contamination": "Exact semantic checks among new groups, not fuzzy or pretraining decontamination. Inherited legacy test overlap with new synthetic controls is not certified."},
        "files": {}, "code_hashes": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}}
    validate_training(parts["train"], manifest)
    for name, partitions, inherited in (("decision-v3", parts, source),
            ("transfer-v3", {"train": [], "calibration": [], "development": transfer_dev, "test": final_extra}, transfer)):
        folder = out / name
        folder.mkdir(parents=True, exist_ok=False)
        m = copy.deepcopy(manifest)
        if name.startswith("transfer"):
            m.update(trainable_sources=[], holdout_sources=manifest["eval_only_sources"], eval_only=True)
        for split, records in partitions.items():
            path = folder / f"{split}.jsonl"
            payload = b""
            inherited_records = inherited_questions = 0
            if split == "test":
                test_src = Path(inherit_eval) if (inherit_eval and name.startswith("decision")) else inherited
                old = read_manifest(test_src)["files"]["test.jsonl"]
                if digest(test_src / "test.jsonl") != old["sha256"]:
                    raise ValueError("inherited test checksum mismatch")
                payload = (test_src / "test.jsonl").read_bytes()
                inherited_records, inherited_questions = old["records"], old["questions"]
                if inherit_eval and name.startswith("decision"): records = []   # the inherited test already contains the locked composition groups
            payload += "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode()
            path.write_bytes(payload)
            m["files"][path.name] = {"sha256": digest(path), "records": inherited_records + len(records),
                                     "questions": inherited_questions + sum(len(r["questions"]) for r in records)}
        write_json(folder / "manifest.json", m)
        print(name, {s: v["records"] for s, v in m["files"].items()}, flush=True)
    if any(digest(p) != h for p, h in parent_hashes.items()):
        raise ValueError("parent artifacts changed")
    return manifest


def smoke_subset(source, out):
    source, out = Path(source), Path(out)
    manifest = read_manifest(source)
    out.mkdir(parents=True, exist_ok=False)
    manifest["files"] = {}
    for split in SPLITS:
        records = []
        if split != "test":
            groups = defaultdict(list)
            for r in load_split(source, split):
                groups[(r["_meta"]["source"], r["_meta"]["group_id"])].append(r)
            taken = Counter()
            for (name, _), group in groups.items():
                if taken[name] < 2:
                    records.extend(group); taken[name] += 1
        path = out / f"{split}.jsonl"
        write_jsonl(path, records)
        manifest["files"][path.name] = {"sha256": digest(path), "records": len(records),
                                        "questions": sum(len(r["questions"]) for r in records)}
    manifest["smoke_only"] = True
    write_json(out / "manifest.json", manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-from")
    ap.add_argument("--public-train", help="frozen suite whose train/calibration public partitions replace the source's (larger pool)")
    ap.add_argument("--synthetic-scale", type=int, default=1, help="multiply synthetic training pairs/groups per arm (both arms stay equal)")
    ap.add_argument("--inherit-eval", help="decision suite whose development/test bytes are reused verbatim")
    ap.add_argument("--random-structures", type=int, default=0, help="compositional arm from N random rule trees (held-out structures excluded)")
    ap.add_argument("--groups-per-structure", type=int, default=8)
    ap.add_argument("--train-styles", default="0,1", help="rendering styles for training compositional records (2 is locked-test only)")
    ap.add_argument("--unmatched-arms", action="store_true", help="allow the two synthetic arms to differ in size")
    ap.add_argument("--legacy-families", default="v3", choices=["v3", "all"], help="'all' adds the ordinal Score threshold families (deadline stays held out)")
    ap.add_argument("--structures-seed", help="reuse another version's random structures (e.g. v7) so only the legacy arm differs")
    a = ap.parse_args()
    styles = tuple(int(x) for x in a.train_styles.split(","))
    if 2 in styles: ap.error("rendering style 2 is reserved for the locked test")
    if a.smoke_from:
        smoke_subset(a.smoke_from, a.out)
    else:
        freeze(a.out, public_train=a.public_train, synthetic_scale=a.synthetic_scale, inherit_eval=a.inherit_eval,
               random_structures=a.random_structures, groups_per_structure=a.groups_per_structure, train_styles=styles, matched_arms=not a.unmatched_arms,
               legacy_families=FAMILIES + ORDINAL_FAMILIES if a.legacy_families == "all" else FAMILIES, structures_seed=a.structures_seed)


if __name__ == "__main__":
    main()
