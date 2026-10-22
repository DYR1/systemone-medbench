#!/usr/bin/env python3
"""Validate exact Jev inputs, detached references and semantic mappings without a model."""
import argparse
import collections
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import rows,sha,jev_hash,full_credit,save
ARMS={'choice_letters','choice_numbers','choice_shuffled','binary_ovr','choice_none','choice_abstain'}

def validate(data):
    m=json.loads((data/'manifest.json').read_text()); counts=collections.Counter(); total=0; parents=0
    seen_datasets=set()
    for e in m['datasets']:
        ds=e['dataset']; assert ds not in seen_datasets; seen_datasets.add(ds)
        assert ds!='sepsis_physionet40'
        root=data/e['path']; assert root.resolve().is_relative_to(data.resolve())
        assert sha(root/'answers.jsonl')==e['answers_file_sha256']
        gold={r['id']:r['gold'] for r in rows(root/'answers.jsonl')}
        assert len(gold)==e['count'];parents+=len(gold)
        original={r['parent_id']:r for r in rows(root/'choice_letters/jev_requests.jsonl')}
        assert set(original)==set(gold)
        for s in e['variants']:
            arm=s['variant'];assert arm in ARMS
            if arm in ['choice_none','choice_abstain']: assert e['category'].startswith('B')
            for fn,h in s['file_sha256'].items(): assert sha(root/arm/fn)==h,(ds,arm,fn,'hash')
            rs=list(rows(root/arm/'jev_requests.jsonl'));byid={r['id']:r for r in rs}
            assert len(byid)==len(rs)==s['request_count']
            maps=list(rows(root/arm/'mappings.jsonl')); assert len(maps)==s['parent_case_count']
            assert len({r['parent_id'] for r in maps})==len(maps)
            for r in rs:
                assert set(r['body'])=={'model','state','questions'}
                assert jev_hash(r['body'])==r['request_sha256']
                assert list(r['body']['questions'])==['decision']
                q=r['body']['questions']['decision']; assert q['type'] in ['choice','noul']
                assert q.get('instructions') and q.get('criteria')
                assert r['parent_id'] in gold
                if arm=='binary_ovr':
                    assert q['type']=='noul' and set(q['criteria'])=={'true','false'}
                    assert all(isinstance(v,str) and v.strip() for v in q['criteria'].values())
                else: assert q['type']=='choice'
            for mp in maps:
                pid=mp['parent_id']; assert pid in original
                base=original[pid]['body']['questions']['decision']['criteria']
                if arm=='binary_ovr':
                    assert set(mp['option_to_query_id'])==set(base)
                    assert all(qid in byid and byid[qid]['parent_id']==pid for qid in mp['option_to_query_id'].values())
                else:
                    r=byid[pid]; criteria=r['body']['questions']['decision']['criteria']
                    align=mp.get('display_to_canonical',{k:k for k in criteria})
                    assert set(align)==set(criteria) and len(set(align.values()))==len(align)
                    if arm in ['choice_letters','choice_numbers','choice_shuffled']:
                        assert set(align.values())==set(base)
                        assert all(criteria[k]==base[v] for k,v in align.items())
                    if arm=='choice_numbers': assert list(criteria)==[str(i+1) for i in range(len(criteria))]
                    if arm=='choice_none': assert len(base)>3 and len(full_credit(gold[pid]))==1
            total+=len(rs);counts[arm]+=len(rs)
    assert len(seen_datasets)==m['dataset_count']
    assert parents==m['unique_case_count'] and total==m['request_count']
    return {'valid':True,'scope':m['scope'],'datasets':len(seen_datasets),'parents':parents,'requests':total,
            'variants':dict(counts),'manifest_sha256':sha(data/'manifest.json')}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path);a=p.parse_args();r=validate(a.data)
    if a.output:save(a.output,r)
    print(json.dumps(r,indent=2))
