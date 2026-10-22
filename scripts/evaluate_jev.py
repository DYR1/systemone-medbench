#!/usr/bin/env python3
"""Portable Jev API runner. Defaults to offline dry-run; never reads gold labels."""
import argparse
import concurrent.futures as cf
import datetime
import fcntl
import getpass
import http.client
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request
from common import ROOT, digest, jev_hash, parse_jev, rows, save, sha

PROVIDERS = {
    'openrouter': ('https://openrouter.ai/api/v1/systemone', 'OPENROUTER_API_KEY',
                   'typesafe/jev-1.13', 'typesafe/jev-1.13-20260917'),
    'typesafe': ('https://api.typesafe.ai/v1/systemone', 'TYPESAFE_API_KEY',
                 'jev-1.13.0', 'jev-1.13.0'),
}
PROFILES = {
    'jev-1.13': ('openrouter', 'typesafe/jev-1.13', 'typesafe/jev-1.13-20260917'),
    'jev-1.14': ('openrouter', 'typesafe/jev-1.14', 'typesafe/jev-1.14-20261014'),
    'microsoft-decision-1': ('decisions', 'microsoft/microsoft-decision-1', 'microsoft/microsoft-decision-1-20261009'),
    'd1-ultra': ('decisions', 'liquid/d1-ultra', 'liquid/d1-ultra-20261015'),
    'gpt6-luna-decisions': ('decisions', 'openai/gpt-6-luna-decisions', 'openai/gpt-6-luna-decisions-20261006'),
}
PROVIDERS['decisions'] = ('https://openrouter.ai/api/alpha/decisions', 'OPENROUTER_API_KEY', None, None)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def selected(args):
    m = json.loads((args.data / 'manifest.json').read_text())
    names = {e['dataset'] for e in m['datasets']}
    arms = {s['variant'] for e in m['datasets'] for s in e['variants']}
    if args.datasets and set(args.datasets) - names:
        raise ValueError('Unknown dataset')
    if args.variants and set(args.variants) - arms:
        raise ValueError('Unknown variant')
    for e in m['datasets']:
        if args.datasets and e['dataset'] not in args.datasets:
            continue
        for s in e['variants']:
            if s['variant'] == 'binary_abstain':
                continue
            if s['variant'] in ('choice_none', 'choice_abstain') and not e['category'].startswith('B'):
                continue
            if args.variants and s['variant'] not in args.variants:
                continue
            p = args.data / e['path'] / s['variant'] / 'jev_requests.jsonl'
            if sha(p) != s['file_sha256'][p.name]:
                raise ValueError('Request file hash mismatch')
            parents = set()
            for r in rows(p):
                if args.case_limit and r['parent_id'] not in parents and len(parents) >= args.case_limit:
                    continue
                parents.add(r['parent_id'])
                if jev_hash(r['body']) != r['request_sha256']:
                    raise ValueError('Request hash mismatch')
                yield e['dataset'], s['variant'], r


def main(args):
    if args.profile:
        args.provider, args.model, args.expected_response_model = PROFILES[args.profile]
    if args.workers < 1 or args.rps <= 0 or args.timeout <= 0 or (args.case_limit is not None and args.case_limit < 1):
        raise ValueError('Positive workers/rps/timeout/case-limit required')
    counts = {}
    for ds, arm, _ in selected(args):
        key = ds + '/' + arm; counts[key] = counts.get(key, 0) + 1
    if not counts:
        raise ValueError('No matching requests')
    print(json.dumps({'mode': 'execute' if args.execute else 'dry_run',
                      'planned_requests': sum(counts.values()), 'arms': counts}, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    if not args.allow_external:
        raise ValueError('Pass --allow-external after checking data licenses and provider processing terms')
    manifest = json.loads((args.data/'manifest.json').read_text())
    if manifest.get('contains_restricted') and not args.allow_restricted:
        raise ValueError('Restricted local data require separate --allow-restricted authorization; this flag does not grant data rights')
    endpoint, env, default_model, default_response = PROVIDERS[args.provider]
    secret = os.environ.get(env, '').strip()
    if not secret:
        secret = getpass.getpass(env + ' (hidden, not saved): ').strip()
    if not secret or any(ord(c) < 33 or ord(c) > 126 for c in secret):
        raise ValueError('Set a valid ' + env + '; keys are never command-line arguments or saved')
    model = args.model or default_model
    expected = args.expected_response_model or default_response
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'run.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {'provider': args.provider, 'wire_model': model, 'expected_response_model': expected,
              'manifest_sha256': sha(args.data / 'manifest.json'), 'datasets': args.datasets,
              'variants': args.variants, 'case_limit': args.case_limit, 'script_sha256': sha(Path(__file__)),
              'common_sha256': sha(Path(__file__).with_name('common.py')),
              'scope': manifest['scope'], 'planned_requests': sum(counts.values()),
              'endpoint': endpoint,
              'probability_source': 'jev_native', 'no_truncation': True}
    config['run_hash'] = digest(config)
    cfg = args.output / 'run_config.json'
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise ValueError('Run configuration changed; use a fresh output directory')
    save(cfg, config)
    prior = {}
    for p in args.output.glob('*/*/events.jsonl'):
        for e in rows(p):
            if e['run_hash'] != config['run_hash']:
                raise ValueError('Mixed run hashes')
            prior[e['dataset'], e['variant'], e['id']] = e
    stop = threading.Event(); pace_lock = threading.Lock(); next_time = 0.0

    def invoke(job):
        nonlocal next_time
        ds, arm, r = job
        body = {**r['body'], 'model': model}
        if args.provider == 'decisions':
            body['provider'] = {'allow_fallbacks': False, 'data_collection': 'deny'}
        event = {'dataset': ds, 'variant': arm, 'id': r['id'], 'parent_id': r['parent_id'],
                 'request_sha256': r['request_sha256'], 'wire_request_sha256': jev_hash(body),
                 'run_hash': config['run_hash'], 'transport_provider': args.provider,
                 'wire_model': model, 'probability_source': 'jev_native', 'full_input_sent': True,
                 'request': r['body'], 'wire_request': body}
        attempts = []
        request_started = time.monotonic()
        for attempt in range(3):
            with pace_lock:
                pause = max(0, next_time - time.monotonic())
                next_time = max(next_time, time.monotonic()) + 1 / args.rps
            if stop.wait(pause):
                return None
            code = None
            try:
                request = urllib.request.Request(endpoint,
                    data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode(),
                    headers={'Authorization': 'Bearer ' + secret, 'Content-Type': 'application/json'}, method='POST')
                with urllib.request.build_opener(NoRedirect()).open(request, timeout=args.timeout) as response:
                    raw = response.read(4 * 1024 * 1024 + 1)
                    if len(raw) > 4 * 1024 * 1024:
                        raise ValueError('Oversized response')
                response = json.loads(raw)
                probabilities, prediction = parse_jev(response, body['questions']['decision'], expected)
                event.update(status='ok', response=response, probabilities=probabilities, prediction=prediction)
                break
            except Exception as exc:
                if isinstance(exc, urllib.error.HTTPError):
                    code = exc.code
                    message = f'HTTP {code}: ' + exc.read(8192).decode(errors='replace')
                else:
                    message = str(exc)
                message = message.replace(secret, '[REDACTED]')[:1000]
                attempts.append({'attempt': attempt + 1, 'http_status': code, 'error': message})
                length = code == 413 or (code in (400, 422) and bool(re.search(
                    r'max_tokens_exceeded|context.{0,10}length|too (long|many tokens)|token limit', message, re.I)))
                fatal = code in (401, 402, 403, 404) or 'response model' in message
                if fatal:
                    stop.set()
                transient = code in (429, 500, 502, 503, 504, 520, 529) or isinstance(exc, (TimeoutError, urllib.error.URLError, ConnectionError, http.client.IncompleteRead))
                if transient and not fatal and not length and attempt < 2 and not stop.wait(2 ** attempt):
                    continue
                event.update(status='error', http_status=code, error=message, fatal=fatal,
                             error_kind='server_input_too_long' if length else 'request_or_response_error')
                break
        event.update(attempt_errors=attempts, elapsed_seconds=time.monotonic()-request_started,
                     finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        return event

    def pending():
        for job in selected(args):
            ds, arm, r = job; old = prior.get((ds, arm, r['id']))
            if old:
                if old['request_sha256'] != r['request_sha256']:
                    raise ValueError('Changed request during resume')
                if old['status'] == 'ok' or (old.get('error_kind') == 'server_input_too_long' and not args.retry_length_errors):
                    continue
            yield job

    work = iter(pending()); active = {}; handles = {}; started = time.monotonic()
    def progress(status):
        save(args.output / 'progress.json', {'status': status, 'planned_requests': sum(counts.values()),
             'successful': sum(e['status'] == 'ok' for e in prior.values()),
             'errors': sum(e['status'] != 'ok' for e in prior.values()),
             'elapsed_seconds': time.monotonic() - started})
    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            exhausted = False
            while active or not exhausted:
                while not stop.is_set() and not exhausted and len(active) < args.workers:
                    job = next(work, None)
                    if job is None:
                        exhausted = True; break
                    active[pool.submit(invoke, job)] = job
                if not active:
                    break
                done, _ = cf.wait(active, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    active.pop(future); e = future.result()
                    if e is None:
                        continue
                    key = (e['dataset'], e['variant']); path = args.output / key[0] / key[1] / 'events.jsonl'
                    if key not in handles:
                        path.parent.mkdir(parents=True, exist_ok=True); handles[key] = path.open('a')
                    handles[key].write(json.dumps(e, ensure_ascii=False, allow_nan=False) + '\n'); handles[key].flush()
                    prior[(key[0], key[1], e['id'])] = e
                progress('running')
        progress('stopped_on_fatal_error' if stop.is_set() else 'finished')
    finally:
        stop.set()
        for f in handles.values():
            f.close()
        lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data/jev')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--provider', choices=PROVIDERS, default='openrouter')
    p.add_argument('--profile', choices=PROFILES)
    p.add_argument('--model'); p.add_argument('--expected-response-model')
    p.add_argument('--datasets', nargs='+'); p.add_argument('--variants', nargs='+')
    p.add_argument('--case-limit', type=int)
    p.add_argument('--workers', type=int, default=4); p.add_argument('--rps', type=float, default=2)
    p.add_argument('--timeout', type=float, default=45)
    p.add_argument('--retry-length-errors', action='store_true')
    p.add_argument('--execute', action='store_true'); p.add_argument('--allow-restricted', action='store_true')
    p.add_argument('--allow-external', action='store_true')
    main(p.parse_args())
