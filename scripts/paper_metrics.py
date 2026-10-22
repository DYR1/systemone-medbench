import hashlib
import statistics
from pathlib import Path
from common import rows, sha, full_credit
from score_core import metrics, compare, threshold_passes
DATA = Path(__file__).resolve().parents[1]/'data/jev'
OUTPUT = Path('.')
ARMS = ['choice_letters','choice_numbers','choice_shuffled','binary_ovr']
THRESHOLDS = [.70,.75,.80,.85,.90]
PCTS = [5,10,15,20,25]

def included(e, arm):
    return arm in ARMS or (e['category'].startswith('B') and arm in ['choice_none','choice_abstain'])

def avg(xs):
    xs=[x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None

def hit(r): return r['prediction'] in full_credit(r['gold'])

def acc(rs): return avg([hit(r) for r in rs])

def category(e): return 'D' if e['category'][0]=='E' else e['category'][0]

def fmt(v): return '—' if v is None else f'{v*100:.2f}'

def table(headers, rr):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,r))+' |' for r in rr])

def reference(manifest):
    refs={}
    for e in manifest['datasets']:
        ds=e['dataset']; root=DATA/e['path']
        original={r['id']:r['gold'] for r in rows(root/'answers.jsonl')}
        labels=list(next(rows(root/'choice_letters/mappings.jsonl'))['canonical_to_display'])
        refs[ds]={'original':original,'labels':labels,'variants':{}}
        for s in e['variants']:
            a=s['variant']
            if not included(e,a): continue
            for fn,h in s['file_sha256'].items():
                assert sha(root/a/fn)==h, (ds,a,fn,'changed input')
            mappings={r['parent_id']:r for r in rows(root/a/'mappings.jsonl')}
            golds={i:original[i] for i in mappings}
            if a=='choice_none': golds={r.get('parent_id',r['id']):r['gold'] for r in rows(root/a/'answers.jsonl')}
            refs[ds]['variants'][a]={'mappings':mappings,'golds':golds,'root':root/a}
    return refs

def summarize(cases, manifest, refs):
    out={'datasets':{},'categories':{},'thresholds':{},'global_top':{},'distributions':{}}
    for e in manifest['datasets']:
        ds=e['dataset']; arms=cases[ds]; result={'category':category(e),'n':e['count'],'classes':len(refs[ds]['labels']),'variants':{},'robustness':{}}
        for a,rs in arms.items():
            m=metrics(refs[ds]['variants'][a]['golds'],rs,refs[ds]['labels'],allow_f1=a!='choice_none')
            result['variants'][a]={'accuracy':m['reference_hit_accuracy_nonabstained'],
                'macro_f1':m['macro_f1_nonabstained_unique_gold'],'n':m['valid_n'],
                'retained_n':m['answered_n'],'f1_n':m['f1_n']}
        for a in ARMS[1:]:
            cm,_=compare(arms['choice_letters'],arms[a],refs[ds]['original'])
            result['robustness'][a]={'accuracy_delta_pp':100*(acc(list(arms[a].values()))-acc(list(arms['choice_letters'].values()))),**cm}
        result['accuracy_variance_pp2']=statistics.pvariance([100*result['variants'][a]['accuracy'] for a in ARMS])
        if 'choice_none' in arms:
            rr=arms['choice_none']; orig={i:arms['choice_letters'][i] for i in rr}; good=[i for i in rr if hit(orig[i])]
            result['none']={'original_accuracy':acc(list(orig.values())),'none_accuracy':acc(list(rr.values())),
                'delta_pp':100*(acc(list(rr.values()))-acc(list(orig.values()))),
                'slot_flip':avg([orig[i]['prediction']!=rr[i]['prediction'] for i in rr]),
                'correctness_flip':avg([hit(orig[i])!=hit(rr[i]) for i in rr]),
                'correct_to_incorrect_rate':avg([hit(orig[i]) and not hit(rr[i]) for i in rr]),
                'incorrect_to_correct_rate':avg([not hit(orig[i]) and hit(rr[i]) for i in rr]),
                'conditional_none_accuracy':acc([rr[i] for i in good]),'original_correct_n':len(good),'n':len(rr)}
        if 'choice_abstain' in arms:
            rr=arms['choice_abstain']; retained={i:r for i,r in rr.items() if not r['abstained']}
            a0=acc(list(arms['choice_letters'].values())); aS0=acc([arms['choice_letters'][i] for i in retained]); aS=acc(list(retained.values()))
            result['abstain']={'original_accuracy':a0,'same_subset_original_accuracy':aS0,'retained_accuracy':aS,
                'abstention_rate':1-len(retained)/len(rr),'retained_n':len(retained),'n':len(rr),
                'delta_pp':100*(aS-a0) if aS is not None else None,
                'same_subset_delta_pp':100*(aS-aS0) if aS is not None else None}
        out['datasets'][ds]=result
    for c in ['A','B','C','D','Overall']:
        ds=[d for d,r in out['datasets'].items() if c=='Overall' or r['category']==c]
        out['categories'][c]={a:{k:avg([out['datasets'][d]['variants'][a][k] for d in ds]) for k in ['accuracy','macro_f1']} for a in ARMS}
    out['robustness']={a:{k:avg([r['robustness'][a][k] for r in out['datasets'].values()])
        for k in ['accuracy_delta_pp','flipped','tv','mean_absolute_probability_difference','js_bits']} for a in ARMS[1:]}
    out['robustness']['mean_reversal']=avg([out['robustness'][a]['flipped'] for a in ARMS[1:]])
    out['robustness']['accuracy_variance_pp2']=avg([r['accuracy_variance_pp2'] for r in out['datasets'].values()])
    for name in ['none','abstain']:
        rr=[r[name] for r in out['datasets'].values() if name in r]
        out[name]={k:avg([r[k] for r in rr]) for k in rr[0] if not k.endswith('_n') and k!='n'} if rr else {}
        out[name]['total_n']=sum(r['n'] for r in rr)
    def selection(rs):
        return {'n':len(rs),'accuracy':acc(rs),'coverage':len(rs)/manifest['unique_case_count'],
            'by_dataset':{e['dataset']:{'n':sum(r['dataset']==e['dataset'] for r in rs),
                'coverage':sum(r['dataset']==e['dataset'] for r in rs)/e['count'],
                'accuracy':acc([r for r in rs if r['dataset']==e['dataset']])} for e in manifest['datasets']}}
    for a in ['choice_letters','binary_ovr']:
        rs=[dict(r,dataset=ds,key=hashlib.sha256(('fixed-coverage-v1\0'+ds+'\0'+i).encode()).hexdigest()) for ds,arms in cases.items() for i,r in arms[a].items()]
        assert len(rs)==manifest['unique_case_count']
        out['thresholds'][a]={str(t):selection([r for r in rs if threshold_passes(r['confidence'],t)]) for t in THRESHOLDS}
        ordered=sorted(rs,key=lambda r:(-r['confidence'],r['key']))
        out['global_top'][a]={str(p):selection(ordered[:len(rs)*p//100]) for p in PCTS}
        out['distributions'][a]={'mean':avg([r['confidence'] for r in rs]),
            'std':statistics.pstdev([r['confidence'] for r in rs]),'exact_one':sum(r['confidence']==1 for r in rs),
            'errors_at_one':sum(r['confidence']==1 and not hit(r) for r in rs),
            'histogram_width_005':[sum(min(19,int(r['confidence']*20))==b for r in rs) for b in range(20)]}
    return out

def render(models,manifest):
    names=list(models)
    def delta(x): return '—' if x is None else f'{x:+.2f}'
    lines=['# Medical decision reliability results','',
        f"Scope: **{manifest['scope']}**. {len(manifest['datasets'])} datasets; {manifest['unique_case_count']:,} parent entries.",'',
        'Public-subset or exclusion-cohort results are not the complete paper benchmark. Overall, category, robustness and abstention rates are equally weighted dataset means. Probability selection pools entries within this exact declared scope.','',
        'Accuracy uses accepted references (maximum expert-score options for SCT). Macro-F1 uses uniquely accepted references. Selected-answer probability is used, not a separate API confidence field.','',
        '## 1. Overall performance','',table(['Task family']+names,
        [[c]+[fmt(models[n]['categories'][c]['choice_letters']['accuracy'])+' / '+fmt(models[n]['categories'][c]['choice_letters']['macro_f1']) for n in names] for c in ['A','B','C','D','Overall']]),'',
        'Cells show Accuracy / macro-F1 (%). A: Triage and Navigation; B: Clinical Judgment; C: Safety Review; D: Physiological Assessment.']
    for a in ARMS:
        lines+=['','### '+a,'',table(['Category','Dataset','N']+names,
            [[category(e),e['dataset'],e['count']]+[fmt(models[n]['datasets'][e['dataset']]['variants'][a]['accuracy'])+' / '+fmt(models[n]['datasets'][e['dataset']]['variants'][a]['macro_f1']) for n in names] for e in manifest['datasets']])]
    lines+=['','## 2. Format robustness','',table(['Model','Numeric delta pp','Shuffle delta pp','OBS delta pp','Variance pp²','Numeric reversal %','Shuffle reversal %','OBS reversal %','Mean reversal %'],
        [[n]+[delta(models[n]['robustness'][a]['accuracy_delta_pp']) for a in ARMS[1:]]+[f"{models[n]['robustness']['accuracy_variance_pp2']:.2f}"]+[fmt(models[n]['robustness'][a]['flipped']) for a in ARMS[1:]]+[fmt(models[n]['robustness']['mean_reversal'])] for n in names])]
    lines+=['','## 3. Abstention awareness (eligible B datasets)','',
        table(['Model','Original Acc %','NONE Acc %','Delta pp','Slot flip %','Correctness flip %','Conditional NONE Acc %'],
        [[n]+[fmt(models[n]['none'].get(k)) for k in ['original_accuracy','none_accuracy']]+[delta(models[n]['none'].get('delta_pp'))]+[fmt(models[n]['none'].get(k)) for k in ['slot_flip','correctness_flip','conditional_none_accuracy']] for n in names]),'',
        table(['Model','Original full Acc %','Original same-subset Acc %','Abstention %','Retained Acc %','Full-cohort delta pp','Same-subset delta pp'],
        [[n]+[fmt(models[n]['abstain'].get(k)) for k in ['original_accuracy','same_subset_original_accuracy','abstention_rate','retained_accuracy']]+[delta(models[n]['abstain'].get(k)) for k in ['delta_pp','same_subset_delta_pp']] for n in names])]
    lines+=['','## 4. Probability-based reliability','','Cells show retained Accuracy % / N / pooled coverage %. No selective-subset F1.']
    for kind,title,values in [('thresholds','Probability thresholds',THRESHOLDS),('global_top','Matched global coverage (%)',PCTS)]:
        rr=[]
        for n in names:
            for arm in ['choice_letters','binary_ovr']:
                rr.append([n,'Choice' if arm=='choice_letters' else 'OBS']+[fmt(models[n][kind][arm][str(v)]['accuracy'])+' / '+str(models[n][kind][arm][str(v)]['n'])+' / '+fmt(models[n][kind][arm][str(v)]['coverage']) for v in values])
        lines+=['','### '+title,'',table(['Model','Interface']+[str(v) for v in values],rr)]
    lines+=['','See comparison.json for all per-dataset metrics, TV, selected-set composition and probability distributions. See audit.json for exact scope, exclusions, inputs and run fingerprints.']
    (OUTPUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
