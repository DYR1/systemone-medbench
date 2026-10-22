"""Portable, standard-library-only I/O and probability validation."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def jev_hash(body):
    # Exact historical serialization; distinct from the LLM canonical hash.
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def normalized(p, keys=None, tolerance=None):
    if not isinstance(p, dict) or not p or (keys is not None and set(p) != set(keys)):
        raise ValueError('Missing, extra or empty probability labels')
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
               and math.isfinite(x) and 0 <= x <= 1 for x in p.values()):
        raise ValueError('Invalid probability value')
    total = math.fsum(p.values())
    if total <= 0 or (tolerance is not None and abs(total - 1) > tolerance):
        raise ValueError('Invalid probability sum')
    return {k: v / total for k, v in p.items()}


def parse_jev(response, question, expected_model=None):
    if not isinstance(response, dict):
        raise ValueError('Response is not an object')
    if expected_model is not None and response.get('model') != expected_model:
        raise ValueError('Different or missing response model')
    answers = response.get('answers')
    if not isinstance(answers, dict) or set(answers) != {'decision'}:
        raise ValueError('Expected exactly answers.decision')
    answer = answers['decision']
    if answer.get('type') != question['type']:
        raise ValueError('Wrong response type')
    if question['type'] == 'noul':
        q = answer['noul']
        if isinstance(q, bool) or not isinstance(q, (float, int)) or not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError('Invalid noul probability')
        p = {'A': q, 'B': 1 - q}
        return p, max(p, key=p.get)
    if question['type'] != 'choice':
        raise ValueError('Only choice/noul are in this release')
    p = normalized(answer['probabilities'], question['criteria'], tolerance=.01)
    if answer['choice'] not in p:
        raise ValueError('Unknown choice label')
    return p, answer['choice']


def full_credit(gold):
    if 'label' in gold:
        return {gold['label']}
    if 'acceptable_labels' in gold:
        return set(gold['acceptable_labels'])
    scores = gold['option_scores']; top = max(scores.values())
    return {k for k, v in scores.items() if v == top}


def credit(gold, prediction):
    if 'option_scores' in gold:
        return gold['option_scores'].get(prediction, 0.0)
    return float(prediction in full_credit(gold))
