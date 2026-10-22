#!/usr/bin/env python3
"""Audit saved responses and produce report-aligned metrics. Never invokes a model."""
import argparse
import collections
import json
from pathlib import Path
from common import ROOT, digest, jev_hash, normalized, parse_jev, rows, save, sha
from score_core import vector
import paper_metrics as pm

def report(a):
    data=a.data or ROOT/'data/jev'
    m=json.loads((data/'manifest.json').read_text()); pm.DATA=data;pm.OUTPUT=a.output
    refs=pm.reference(m); expected={}; specs={}
    for e in m['datasets']:
        for s in e['variants']:
            ds,arm=e['dataset'],s['variant'];specs[ds,arm]=s
            fn='jev_requests.jsonl'
            for r in rows(data/e['path']/arm/fn):
                q=r['body']['questions']['decision']
                expected[ds,arm,r['id']]={'hash':r['request_sha256'],'parent':r['parent_id'],
                    'question':q}
    assert len(expected)==m['request_count']
    configs={}; files=[]; issues=[]; owners={}
    if a.interface=='native':
        plan=json.loads((a.run/'inputs/plan.json').read_text())
        assert plan['manifest_sha256']==sha(data/'manifest.json')
        protocol_signatures=set(); input_ids=set()
        for s in plan['shards']:
            inp=a.run/'inputs'/f"shard{s['shard']:02d}.jsonl"
            assert sha(inp)==s['sha256']
            for r in rows(inp):
                k=r['dataset'],r['variant'],r['id']
                assert k not in input_ids and k in expected;input_ids.add(k)
                assert jev_hash(r['body'])==r['request_sha256']==expected[k]['hash']
            folder=a.run/'runs'/f"shard{s['shard']:02d}";cp=folder/'config.json'
            if not cp.exists():issues.append({'shard':s['shard'],'error':'missing config'});continue
            cfg=json.loads(cp.read_text());assert cfg['input_sha256']==s['sha256']
            assert cfg['manifest_sha256']==sha(data/'manifest.json')
            assert digest({k:v for k,v in cfg.items() if k!='run_hash'})==cfg['run_hash']
            protocol_signatures.add(digest({k:v for k,v in cfg.items() if k not in ['run_hash','input_sha256']}))
            files.append((folder/'events.jsonl',cfg))
        assert len(protocol_signatures)<=1,'Mixed checkpoints, code, or protocols across shards'
        assert input_ids==set(expected),'Shard plan does not cover exact report scope'
    else:
        cfg=json.loads((a.run/'run_config.json').read_text())
        assert cfg['manifest_sha256']==sha(data/'manifest.json')
        for ds,arm in specs: files.append((a.run/ds/arm/'events.jsonl',cfg))
    latest={}; event_counts=collections.Counter(); source_hashes={}
    for path,cfg in files:
        configs[cfg['run_hash']]=cfg
        if not path.exists():continue
        source_hashes[str(path.relative_to(a.run))]=sha(path)
        for ev in rows(path):
            k=ev['dataset'],ev['variant'],ev['id'];assert k in expected,('unknown result',k)
            assert k not in owners or owners[k]==str(path),'Same request appears in different shards/files'
            owners[k]=str(path);r=expected[k]
            assert ev['run_hash']==cfg['run_hash'] and ev['request_sha256']==r['hash']
            assert ev['parent_id']==r['parent'];event_counts[k]+=1
            if ev['status']!='ok':
                latest[k]={'status':'error','error':ev.get('error_kind',ev.get('error','request error'))};continue
            if a.interface=='native':assert jev_hash(ev['request'])==r['hash']
            p,pred=parse_jev(ev['response'],r['question'],cfg.get('expected_response_model',cfg.get('model_name')))
            assert set(p)==set(ev['probabilities']) and all(abs(v-ev['probabilities'][k])<1e-12 for k,v in p.items()) and pred==ev['prediction'],'Response/parsed probability disagreement'
            latest[k]={'status':'ok','probabilities':p,'prediction':pred}
    cases={}; audits=[]
    for e in m['datasets']:
        ds=e['dataset'];cases[ds]={}
        for arm,ref in refs[ds]['variants'].items():
            keys=[k for k in expected if k[:2]==(ds,arm)]
            parsed={k[2]:latest[k] for k in keys if latest.get(k,{}).get('status')=='ok'}
            for k in keys:
                if k[2] not in parsed:issues.append({'dataset':ds,'variant':arm,'id':k[2],
                    'error':latest.get(k,{}).get('error','missing response')})
            vs={}
            for pid,mp in ref['mappings'].items():
                try: v=vector(mp,parsed,arm)
                except (KeyError,ValueError) as ex:
                    issues.append({'dataset':ds,'variant':arm,'parent_id':pid,'error':str(ex)});continue
                vs[pid]={'id':pid,'gold':ref['golds'][pid],**v}
            cases[ds][arm]=vs
            audits.append({'dataset':ds,'variant':arm,'planned_requests':len(keys),'successful_requests':len(parsed),
                'planned_parents':len(ref['mappings']),'valid_parents':len(vs)})
    audit={'complete':not issues,'issues':issues,'arms':audits,'planned_requests':len(expected),
           'successful_requests':sum(v['status']=='ok' for v in latest.values()),
           'retried_request_ids':sum(n>1 for n in event_counts.values()),'run_configs':configs,
           'response_file_sha256':source_hashes,'manifest_sha256':sha(data/'manifest.json'),
           'scorer_hashes':{p.name:sha(p) for p in [Path(__file__),Path(pm.__file__),ROOT/'scripts/score_core.py',ROOT/'scripts/common.py']}}
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'comparison.json').exists():
        old=json.loads((a.output/'comparison.json').read_text())
        assert old.get('source_run')==str(a.run.resolve()),'Report directory belongs to a different run'
    audit['scope']=m['scope']
    audit['all_requests_attempted']=set(latest)==set(expected)
    audit['excluded_parents']=[]
    if issues and getattr(a,'exclude_length_errors',False):
        allowed={'context_length','server_input_too_long'}
        errors={k:v for k,v in latest.items() if v['status']!='ok'}
        if set(latest)!=set(expected) or not errors or any(v['error'] not in allowed for v in errors.values()):
            raise ValueError('Only explicit length errors may be excluded, after every request has been attempted')
        bad={(ds,expected[(ds,arm,rid)]['parent']) for ds,arm,rid in errors}
        # Every missing vector must be explained by one of those exact affected parents.
        for issue in issues:
            if 'dataset' not in issue: raise ValueError('Unexplained shard/config issue')
            pid=issue.get('parent_id')
            if pid is None:pid=expected[(issue['dataset'],issue['variant'],issue['id'])]['parent']
            if (issue['dataset'],pid) not in bad:raise ValueError('Unexplained invalid result; exclusion refused')
        for e in m['datasets']:
            ds=e['dataset']; removed={pid for d,pid in bad if d==ds}
            for arm in list(cases[ds]):
                cases[ds][arm]={i:r for i,r in cases[ds][arm].items() if i not in removed}
                refs[ds]['variants'][arm]['golds']={i:r for i,r in refs[ds]['variants'][arm]['golds'].items() if i not in removed}
                if not cases[ds][arm]:raise ValueError('Exclusion empties an arm; use a separately declared evaluation scope')
            e['count']-=len(removed)
        audit['excluded_parents']=[{'dataset':ds,'parent_id':pid} for ds,pid in sorted(bad)]
        audit['matched_cohort_valid']=True
        m['unique_case_count']=sum(e['count'] for e in m['datasets'])
        m['scope']+='__matched_length_exclusion_NOT_full_paper'
        audit['scored_scope']=m['scope'];audit['scored_parents']=m['unique_case_count']
    save(a.output/'audit.json',audit)
    if issues and not audit.get('matched_cohort_valid'):
        print(json.dumps({'complete':False,'successful_requests':audit['successful_requests'],'issues':len(issues),
            'message':'No complete performance report generated. Missing requests are not scored as incorrect.'}))
        return False
    summary=pm.summarize(cases,m,refs)
    save(a.output/'comparison.json',{'source_run':str(a.run.resolve()),'scope':m['scope'],
        'models':{a.name:summary},'complete':not issues,'matched_cohort_valid':True,
        'scored_parents':m['unique_case_count'],'excluded_parents':audit['excluded_parents']})
    pm.render({a.name:summary},m)
    with (a.output/'case_results.jsonl').open('w') as f:
        for ds,arms in cases.items():
            for arm,rr in arms.items():
                for row in rr.values():f.write(json.dumps(dict(dataset=ds,variant=arm,**row),ensure_ascii=False)+'\n')
    print(json.dumps({'complete':not issues,'matched_cohort_valid':True,'model':a.name,'output':str(a.output),'requests':len(expected)}))
    return True

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--interface',choices=['jev','native'],required=True)
    p.add_argument('--data',type=Path);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exclude-length-errors',action='store_true',help='Explicitly exclude affected parents across all formats; never excludes missing/OOM/network errors')
    raise SystemExit(0 if report(p.parse_args()) else 2)
