#!/usr/bin/env python3
"""Read-only speculative acceptance and expert statistics. Standard library only.

snapshot --out before.json ; [run evaluation] ; snapshot --out after.json
report --before before.json --after after.json [--hit-stats hit_stats.json]
report also accepts raw Prometheus text files from /metrics.
CPU JSONL contains CPU expert counts and transfer decisions, not cache hits.
"""
import argparse
import collections
import datetime
import json
import math
from pathlib import Path
import re
import statistics
import sys
import urllib.request

PREFIX = 'vllm:spec_decode_'
NAMES = ('num_drafts', 'num_draft_tokens', 'num_accepted_tokens')


def dump(value, path=None):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if path:
        Path(path).write_text(text, encoding='utf-8')
    print(text, end='')


def parse_metrics(text, model, engine):
    result = {}
    pattern = re.compile(r'^(\w[\w:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$')
    for line in text.splitlines():
        m = pattern.match(line)
        if not m or not m[1].startswith(PREFIX):
            continue
        labels = dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', m[2] or ''))
        if labels.get('model_name') != model or (engine is not None and labels.get('engine') != engine):
            continue
        key = (m[1], tuple(sorted(labels.items())))
        if key in result:
            raise ValueError('Duplicate metrics series: ' + m[1])
        value = float(m[3])
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid counter: ' + m[1])
        result[key] = value
    return result


def read_snapshot(path):
    text = Path(path).read_text(encoding='utf-8')
    if text.lstrip().startswith('{'):
        obj = json.loads(text)
        return obj['metrics'], obj
    return text, {}


def acceptance(before, after, model, engine):
    a, am = read_snapshot(before)
    b, bm = read_snapshot(after)
    if am and bm and am.get('url') != bm.get('url'):
        raise ValueError('Snapshots refer to different endpoints')
    a, b = parse_metrics(a, model, engine), parse_metrics(b, model, engine)
    counts = {}
    for name in NAMES:
        metric = PREFIX + name + '_total'
        ka = {k for k in a if k[0] == metric}
        kb = {k for k in b if k[0] == metric}
        if not ka or ka != kb:
            raise ValueError('Missing/changed metrics series: ' + metric)
        differences = [b[k] - a[k] for k in ka]
        if min(differences) < 0:
            raise ValueError('Counter reset/service restart: ' + metric)
        counts[name] = sum(differences)
    d, t, n = (counts[x] for x in NAMES)
    if n > t or (d == 0 and (t or n)):
        raise ValueError('Inconsistent speculative counters')
    return dict(counts=counts, acceptance_rate=n / t if t else None,
                mean_accepted_length_including_bonus=1 + n / d if d else None,
                mean_accepted_draft_length=n / d if d else None,
                scope='All requests between snapshots, including warmups/concurrent clients if present',
                start=am.get('captured_at'), end=bm.get('captured_at'),
                warning='Endpoint counter reset is detected when counters decrease; a restart followed by larger counters cannot be detected from raw counters alone.')


def hit_statistics(path):
    layers = json.loads(Path(path).read_text(encoding='utf-8'))
    required = ('selections', 'hits', 'unique_selections', 'unique_hits')
    totals = collections.Counter()
    for layer, row in layers.items():
        for key in required:
            value = row[key]
            if not isinstance(value, int) or value < 0:
                raise ValueError('Invalid hit count: ' + str(layer))
            totals[key] += value
        if row['hits'] > row['selections'] or row['unique_hits'] > row['unique_selections']:
            raise ValueError('Hit count exceeds denominator')
    return dict(source=str(path), layers=len(layers), totals=dict(totals),
                route_cache_hit_rate=totals['hits']/totals['selections'] if totals['selections'] else None,
                unique_expert_cache_hit_rate=totals['unique_hits']/totals['unique_selections'] if totals['unique_selections'] else None,
                scope='Exactly the supplied hit_stats.json; requires counts measured before cache update. Not automatically aligned to metrics snapshots.')


def cpu_statistics(paths, tokens, since, until):
    # Streaming scan: memory use is proportional to observed layer-count values.
    hist = collections.Counter()
    transfers = routes = count = missing_t = missing_r = truncated = 0
    layers = collections.Counter()
    dropped = {}
    for path in paths:
        with open(path, encoding='utf-8') as stream:
            for line_no, line in enumerate(stream, 1):
                if 'cpu_moe.compute' not in line and 'cpu_moe.trace_flush' not in line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_no}: invalid/incomplete JSONL; use a settled log copy') from exc
                if row.get('event') == 'cpu_moe.trace_flush':
                    pid = (str(path), row.get('pid'))
                    dropped[pid] = max(dropped.get(pid, 0), row.get('dropped', 0))
                    continue
                if row.get('event') != 'cpu_moe.compute':
                    continue
                if tokens is not None and row.get('tokens') != tokens:
                    continue
                stamp = row['wall_ns']
                if (since is not None and stamp < since) or (until is not None and stamp >= until):
                    continue
                count += 1
                layers[row['layer']] += 1
                hist[row['expert_count']] += 1
                transfers += row.get('transfer_count', 0)
                routes += row.get('expert_route_count', 0)
                missing_t += 'transfer_count' not in row
                missing_r += 'expert_route_count' not in row
                truncated += bool(row.get('expert_ids_truncated'))
    def quantile(q):
        rank = max(1, math.ceil(count*q)); cumulative = 0
        for value, freq in sorted(hist.items()):
            cumulative += freq
            if cumulative >= rank:
                return value
    # Median uses the two central order statistics for even-sized populations.
    median = (quantile(.5) + quantile((count//2+1)/count))/2 if count and count%2 == 0 else quantile(.5) if count else None
    return dict(sources=paths, records=count, calls_per_layer=dict(layers), tokens_filter=tokens,
                wall_ns_interval=[since, until],
                cpu_unique_experts_per_record=dict(mean=sum(k*v for k,v in hist.items())/count,
                    median=median, p90=quantile(.9), max=max(hist)) if count else None,
                cpu_route_selections=routes if not missing_r else None,
                selected_transfer_experts=transfers if not missing_t else None,
                mean_selected_transfer_experts=transfers/count if count and not missing_t else None,
                trace_dropped_lifetime=sum(dropped.values()), truncated_records=truncated,
                cache_hit_rate=None,
                warning='CPU counts exclude NPU-owned routes after transfer selection. These records cannot reconstruct initial cache hits. T counts transfer decisions, not confirmed DMA completions. Each record is one logged layer/partition call; no implicit prefill removal or full-step reconstruction. Do not supply overlapping logs. Dropped counts are lifetime totals, not interval totals.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    snap = sub.add_parser('snapshot', help='Read /metrics without submitting inference')
    snap.add_argument('--url', default='http://127.0.0.1:8150/metrics')
    snap.add_argument('--out', required=True)
    report = sub.add_parser('report', help='Offline interval acceptance and optional expert statistics')
    report.add_argument('--before')
    report.add_argument('--after')
    report.add_argument('--model', default='deepseek_v4')
    report.add_argument('--engine', default='0')
    report.add_argument('--hit-stats', help='Old per-layer hit_stats.json (full routing counters)')
    report.add_argument('--cpu-log', nargs='+', help='CPU JSONL files; no overlapping copies')
    report.add_argument('--tokens', type=int, help='Only CPU records of this token count; 8 for current spec7 batch1 verification, NOT a universal decode filter')
    report.add_argument('--since-ns', type=int)
    report.add_argument('--until-ns', type=int)
    report.add_argument('--out')
    args = parser.parse_args()
    if args.action == 'snapshot':
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(args.url, timeout=15) as response:
            metrics = response.read().decode('utf-8')
        dump(dict(format_version=1, captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  url=args.url, metrics=metrics), args.out)
        return
    if bool(args.before) != bool(args.after):
        parser.error('--before and --after must be supplied together')
    if not (args.before or args.hit_stats or args.cpu_log):
        parser.error('Supply metrics snapshots, --hit-stats, or --cpu-log')
    if args.since_ns is not None and args.until_ns is not None and args.since_ns >= args.until_ns:
        parser.error('--since-ns must be less than --until-ns')
    result = {}
    if args.before:
        result['speculative'] = acceptance(args.before, args.after, args.model, args.engine)
    if args.hit_stats:
        result['expert_cache'] = hit_statistics(args.hit_stats)
    else:
        result['expert_cache'] = dict(cache_hit_rate=None, reason='No complete hit/selection counters supplied')
    if args.cpu_log:
        result['cpu'] = cpu_statistics(args.cpu_log, args.tokens, args.since_ns, args.until_ns)
    dump(result, args.out)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        sys.exit(1)
