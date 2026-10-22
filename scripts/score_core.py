import math
from common import credit, full_credit, normalized

def metrics(golds, vectors, labels, allow_f1=True):
    valid = {i: v for i, v in vectors.items() if i in golds}
    answered = {i: v for i, v in valid.items() if not v['abstained']}
    correct = sum(v['prediction'] in full_credit(golds[i]) for i, v in answered.items())
    total = sum(credit(golds[i], v['prediction']) for i, v in answered.items())
    partial = any('option_scores' in g for g in golds.values())
    pairs = [(next(iter(full_credit(golds[i]))), v['prediction']) for i, v in answered.items()
             if len(full_credit(golds[i])) == 1]
    per_class = {}
    if allow_f1 and pairs:
        for label in labels:
            tp = sum(y == label and pred == label for y, pred in pairs)
            fp = sum(y != label and pred == label for y, pred in pairs)
            fn = sum(y == label and pred != label for y, pred in pairs)
            per_class[label] = {'tp': tp, 'fp': fp, 'fn': fn,
                                'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    return {'planned_n': len(golds), 'valid_n': len(valid), 'invalid_or_missing_n': len(golds) - len(valid),
            'answered_n': len(answered), 'abstained_n': len(valid) - len(answered),
            'coverage': len(valid) / len(golds) if golds else None,
            'abstention_rate_valid': 1 - len(answered) / len(valid) if valid else None,
            'primary_metric': 'expert_partial_credit' if partial else 'acceptable_reference_accuracy',
            'primary_score_nonabstained': total / len(answered) if answered else None,
            'reference_hit_accuracy_nonabstained': correct / len(answered) if answered else None,
            'primary_score_all_valid': total / len(valid) if valid else None,
            'primary_score_all_planned_zero_missing': total / len(golds) if golds else None,
            'macro_f1_nonabstained_unique_gold': (sum(v['f1'] for v in per_class.values()) / len(per_class)
                                                if per_class else None),
            'f1_n': len(pairs) if allow_f1 else 0, 'f1_per_class': per_class}

def threshold_passes(confidence, threshold):
    return confidence >= threshold or math.isclose(confidence, threshold, rel_tol=0, abs_tol=1e-12)

def vector(mapping, parsed, arm):
    abstain = mapping.get('none_id', mapping.get('none_display_id')) if arm.endswith('abstain') else None
    if arm in ('binary_ovr', 'binary_abstain'):
        raw = {k: parsed[qid]['probabilities']['A'] for k, qid in mapping['option_to_query_id'].items()}
        p = normalized(raw); pred = max(p, key=p.get)
    else:
        event = parsed[mapping['parent_id']]; displayed = event['probabilities']
        alignment = mapping.get('display_to_canonical', {k: k for k in displayed})
        if set(alignment) != set(displayed) or len(set(alignment.values())) != len(alignment):
            raise ValueError('Invalid display-to-canonical mapping')
        p = {alignment[k]: v for k, v in displayed.items()}
        pred = alignment[event['prediction']]; raw = None
    return {'prediction': pred, 'probabilities': p, 'confidence': p[pred],
            'abstained': abstain is not None and pred == abstain,
            'raw_affirmative_probabilities': raw,
            'argmax_ties': [k for k, v in p.items() if v == max(p.values())],
            'prediction_disagrees_with_argmax': p[pred] < max(p.values()) - 1e-6}

def compare(base, variant, golds):
    pairs = []
    for i in sorted(set(base) & set(variant)):
        a, b = base[i], variant[i]
        if set(a['probabilities']) != set(b['probabilities']):
            raise ValueError('Cannot compare different semantic candidate sets')
        keys = list(a['probabilities']); p, q = a['probabilities'], b['probabilities']
        d = [abs(p[k] - q[k]) for k in keys]
        js = sum(.5 * x * math.log2(x / ((p[k] + q[k]) / 2))
                 for k in keys for x in (p[k], q[k]) if x > 0)
        accepted = full_credit(golds[i])
        pairs.append({'id': i, 'flipped': a['prediction'] != b['prediction'],
                      'tv': sum(d) / 2, 'mean_absolute_probability_difference': sum(d) / len(d),
                      'max_absolute_probability_difference': max(d), 'js_bits': js,
                      'gold_probability_mass_delta': sum(q[k] - p[k] for k in accepted),
                      'baseline_selected_probability_delta': q[a['prediction']] - p[a['prediction']]})
    fields = ('flipped', 'tv', 'mean_absolute_probability_difference', 'max_absolute_probability_difference',
              'js_bits', 'gold_probability_mass_delta', 'baseline_selected_probability_delta')
    return {'paired_n': len(pairs), **{k: sum(p[k] for p in pairs) / len(pairs) if pairs else None for k in fields}}, pairs
