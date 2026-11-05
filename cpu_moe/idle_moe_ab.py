#!/usr/bin/env python3
"""Codex: reversible CPU idle A/B/A/B experiment on the existing MoE setup."""
import collections
import hashlib
import json
import os
import pathlib
import signal
import statistics
import struct
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path('/mnt/share/y50063808/vllm-ascend-workspace')
OUT = pathlib.Path(sys.argv[1])
CARD = int(sys.argv[2])
PORT = int(sys.argv[3])
if CARD == 0:
    raise ValueError('Codex: card 0 is reserved for the user')
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / '.agents/lib'))
import vaws_run_manifest as rm

script = ROOT / 'cpu_moe/run_dspark_pr98_0731.sh'
snapshot = script.read_bytes()
(OUT / 'launch_script_snapshot.sh').write_bytes(snapshot)
manifest = rm.new_manifest(run_type='performance', environment={'host': '141.61.45.105', 'container': 'cpu_dspark', 'port': PORT}, model={'path': '/mnt/share/DeepSeek-V4-Flash-0731'}, topology={'devices': [CARD]}, command=['bash', str(script)], environment_variables={'CARDS':str(CARD),'PORT':str(PORT),'VLLM_CACHE_ROOT':str(OUT / 'compile_cache')}, workspace_snapshot={'launch_sha256': hashlib.sha256(snapshot).hexdigest()})
manifest = rm.transition_status(manifest, 'running')
rm.write_manifest(OUT / 'manifest.json', manifest)

def post(payload):
    # Codex: identical deterministic payload, with host monotonic request boundaries.
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    start = time.monotonic_ns()
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.load(response)
    return start, time.monotonic_ns(), result

def quantiles(values):
    if not values:
        return {'n': 0}
    values = sorted(values)
    return {'n': len(values), 'p50_us': statistics.median(values), 'p90_us': values[int((len(values)-1)*.9)], 'p99_us': values[int((len(values)-1)*.99)]}

def phases(raw, start, end):
    groups = collections.defaultdict(dict)
    flush = None
    with raw.open() as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get('event') == 'cpu_moe.trace_flush':
                flush = item
            if item.get('event') == 'cpu_moe.lifecycle' and start <= item['mono_ns'] <= end:
                groups[item['task_id']][item['phase']] = item
    pairs = [('callback_enter','queue_publish'), ('queue_publish','queue_take'), ('queue_take','compute_begin'), ('compute_begin','compute_end'), ('compute_end','task_return'), ('task_return','completion_publish'), ('completion_publish','completion_observed'), ('completion_observed','drain_return'), ('drain_return','callback_exit')]
    summary = {}
    for a,b in pairs:
        vals = [(g[b]['mono_ns']-g[a]['mono_ns'])/1000 for g in groups.values() if a in g and b in g and g[b]['mono_ns'] >= g[a]['mono_ns']]
        summary[a+' -> '+b] = quantiles(vals)
    overhead = []
    for g in groups.values():
        if all(k in g for k in ['callback_enter','callback_exit','compute_begin','compute_end']):
            overhead.append((g['callback_exit']['mono_ns']-g['callback_enter']['mono_ns']-g['compute_end']['mono_ns']+g['compute_begin']['mono_ns'])/1000)
    summary['callback non-compute'] = quantiles(overhead)
    return {'tasks': len(groups), 'writer': flush, 'phases': summary}

service = None
latency_fd = None
results = []
log = (OUT / 'service.log').open('w')
try:
    # Codex: only this experiment's process group is cleaned up; existing services are untouched.
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2):
            raise RuntimeError(f'Port {PORT} already has a service; refuse to replace it')
    except urllib.error.URLError:
        pass
    # Codex: reserve the user's card 0 and override only card and HTTP port.
    # Codex: isolate compilation artifacts so another service's cache cannot enter this run.
    env = dict(os.environ, CARDS=str(CARD), PORT=str(PORT), VLLM_CACHE_ROOT=str(OUT / 'compile_cache'))
    service = subprocess.Popen(['bash', str(script)], cwd=script.parent, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (OUT / 'service_pid.txt').write_text(str(service.pid))
    print('Codex: service launched pid='+str(service.pid), flush=True)
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if service.poll() is not None:
            raise RuntimeError('Service exited before ready, rc='+str(service.returncode))
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        time.sleep(5)
    else:
        raise TimeoutError('Service readiness timeout')
    profile = (script.parent / 'cpu_profile/latest').resolve()
    raw_files = list(profile.glob('cpu_compute_*.jsonl'))
    if len(raw_files) != 1:
        raise RuntimeError('Expected one CPU trace, got '+str(raw_files))
    raw = raw_files[0]
    (OUT / 'trace_path.txt').write_text(str(raw))
    print('Codex: service ready; trace='+str(raw), flush=True)
    payload = {'model': 'deepseek_v4', 'prompt': 'Hello', 'max_tokens': 64, 'temperature': 0, 'seed': 1024, 'ignore_eos': True}
    (OUT / 'request_payload.json').write_text(json.dumps(payload, indent=2))
    post(payload)
    for name, constrained in [('A1_default',False),('B1_low_latency',True),('A2_default',False),('B2_low_latency',True)]:
        # Codex: the PM QoS request exists only while this descriptor is open.
        # This is a host-wide temporary constraint, not a persistent sysfs change.
        if constrained:
            latency_fd = os.open('/dev/cpu_dma_latency', os.O_RDWR | os.O_CLOEXEC)
            if os.write(latency_fd, struct.pack('i', 0)) != 4:
                raise RuntimeError('PM QoS request write failed')
        time.sleep(2)
        post(payload)
        for index in range(3):
            time.sleep(1)
            start,end,response = post(payload)
            time.sleep(.5)
            lifecycle = phases(raw,start,end)
            item = {'case': name, 'constraint_us': 0 if constrained else None, 'request': index, 'start_mono_ns': start, 'end_mono_ns': end, 'elapsed_s': (end-start)/1e9, 'usage': response.get('usage'), 'text_sha256': hashlib.sha256(response['choices'][0]['text'].encode()).hexdigest(), **lifecycle}
            results.append(item)
            (OUT / (name+'_'+str(index)+'_response.json')).write_text(json.dumps(response,ensure_ascii=False,indent=2))
            (OUT / 'results.json').write_text(json.dumps(results,indent=2))
            selected = {k: v for k,v in lifecycle['phases'].items() if k in ['queue_publish -> queue_take','completion_publish -> completion_observed','callback non-compute']}
            print(json.dumps({'case':name,'request':index,'elapsed_s':item['elapsed_s'],'tasks':item['tasks'],'phases':selected}),flush=True)
        if latency_fd is not None:
            os.close(latency_fd)
            latency_fd = None
    manifest = rm.transition_status(manifest, 'passed')
except BaseException as e:
    print('Codex: experiment failed: '+repr(e),flush=True)
    (OUT / 'error.txt').write_text(repr(e))
    manifest = rm.transition_status(manifest, 'failed')
    raise
finally:
    if latency_fd is not None:
        os.close(latency_fd)
    print('Codex: temporary idle constraint released',flush=True)
    if service is not None:
        try:
            os.killpg(service.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        # Codex: wait for the whole owned process group, not just the launch shell.
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            service.poll()
            try:
                os.killpg(service.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(1)
        else:
            try:
                os.killpg(service.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        service.wait(timeout=15)
    log.close()
    if (OUT / 'results.json').exists():
        manifest = rm.add_artifact(manifest,name='experiment_results',kind='json',uri=str(OUT / 'results.json'))
    rm.write_manifest(OUT / 'manifest.json',manifest)
    (OUT / 'cleanup.json').write_text(json.dumps({'latency_constraint_released':True,'service_returncode':None if service is None else service.returncode}))
