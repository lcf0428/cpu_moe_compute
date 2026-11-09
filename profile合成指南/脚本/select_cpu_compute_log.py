#!/usr/bin/env python3
"""Select a CPU log belonging to the profiler process and time window."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def select_log(trace, cpu_dir):
    trace = Path(trace).resolve()
    match = re.search(r'rank\d+_(\d+)_\d+_ascend_pt$', trace.parent.parent.name)
    pid = int(match[1]) if match else None
    raw = json.loads(trace.read_text(encoding='utf8'))
    events = raw if isinstance(raw, list) else raw['traceEvents']
    stamps = [(float(e['ts']), float(e['ts']) + float(e.get('dur', 0)))
              for e in events if 'ts' in e]
    if not stamps:
        raise ValueError('Ascend trace has no timestamped events')
    low, high = min(a for a, _ in stamps), max(b for _, b in stamps)
    pattern = f'cpu_compute_*_{pid}.jsonl' if pid is not None else 'cpu_compute_*.jsonl'
    candidates = {p.resolve() for directory in (trace.parent.parent.parent, Path(cpu_dir))
                  for p in directory.glob(pattern) if p.is_file()}
    matching = []
    for path in sorted(candidates):
        with path.open(encoding='utf8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get('event') != 'cpu_moe.compute' or (pid is not None and r.get('pid') != pid):
                    continue
                a = int(r['wall_ns']) / 1000
                dur = int(r['duration_ns']) / 1000
                if dur >= 0 and a < high and a + dur > low:
                    matching.append(path)
                    break
    # A saved sidecar can be a byte-identical copy of the container log.
    if len(matching) > 1:
        hashes = []
        for path in matching:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            hashes.append(digest.digest())
        if len(set(hashes)) == 1:
            return min(matching, key=lambda p: (p.parent != trace.parent.parent.parent, str(p)))
    if len(matching) != 1:
        raise ValueError(f'Expected one CPU log matching profiler PID {pid} and time window; '
                         f'found {len(matching)}. Pass the CPU log explicitly or save it in '
                         'the capture directory. Matching files: ' + ', '.join(map(str, matching)))
    return matching[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('cpu_dir', type=Path)
    args = parser.parse_args()
    try:
        print(select_log(args.trace, args.cpu_dir))
    except (ValueError, OSError) as exc:
        parser.exit(2, f'ERROR: {exc}\n')
