#!/usr/bin/env python3
"""Merge actual C++ CPU expert forward spans into an Ascend Chrome trace."""
import argparse
import bisect
import collections
import json
from pathlib import Path
import statistics


def cpu_span_name(record):
    """One-based layer, unique CPU experts, non-deduplicated CPU routes."""
    calls = record.get('expert_route_count')
    return f"L{int(record['layer']) + 1}_E{record['expert_count']}_C{calls if calls is not None else 'unknown'}"


def union(intervals):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def intersect(xs, ys):
    out, i, j = [], 0, 0
    while i < len(xs) and j < len(ys):
        lo, hi = max(xs[i][0], ys[j][0]), min(xs[i][1], ys[j][1])
        if hi > lo:
            out.append((lo, hi))
        if xs[i][1] < ys[j][1]:
            i += 1
        else:
            j += 1
    return out


def overlap(a, b, intervals, ends):
    total = 0.0
    i = bisect.bisect_right(ends, a)
    while i < len(intervals) and intervals[i][0] < b:
        lo, hi = intervals[i]
        total += max(0, min(b, hi) - max(a, lo))
        i += 1
    return total


def pick_anchor(events):
    """The device process and its busiest track, so CPU rows sit next to the ops.

    Ascend Hardware carries the real device timeline -- the ops themselves and
    the waits between them -- which is what CPU overlap has to be read against.
    Overlap Analysis is a derived aggregate, so it is only a fallback.
    """
    names = {e['pid']: e.get('args', {}).get('name') for e in events
             if e.get('ph') == 'M' and e.get('name') == 'process_name'}
    counts = collections.Counter((e.get('pid'), e.get('tid')) for e in events if e.get('ph') == 'X')
    for want in ('Ascend Hardware', 'Overlap Analysis'):
        pids = [p for p, n in names.items() if n == want and any(k[0] == p for k in counts)]
        if not pids:
            continue
        pid = max(pids, key=lambda p: sum(c for k, c in counts.items() if k[0] == p))
        tid = 27 if (pid, 27) in counts else max((k[1] for k in counts if k[0] == pid), key=lambda t: counts[(pid, t)])
        return pid, tid, want
    return None, None, None


def reserve_tids(events, anchor, first_after, parts):
    """Free tids directly above the Computing track, one per NUMA part.

    Sitting just above Computing keeps the rows adjacent even in viewers that
    ignore thread_sort_index, and skipping used tids avoids merging two
    different tracks onto one row.
    """
    used = {e.get('tid') for e in events if e.get('pid') == anchor}
    tid_for, nxt = {}, first_after - 1
    for part in parts:
        while nxt in used:
            nxt -= 1
        tid_for[part] = nxt
        used.add(nxt)
        nxt -= 1
    return tid_for


def reorder_threads(events, anchor, cpu_tids, lead_tids):
    """Renumber the anchor process so the CPU rows precede the lead device track.

    Existing tracks keep their relative order and only the CPU rows are inserted,
    so the rest of the device layout stays as the profiler produced it. Tracks are
    collected from thread metadata as well as from events: a track may carry a
    sort index without ever producing an event, and leaving it out would let a
    stale index tie with the one assigned here.
    """
    existing, known = {}, set()
    for e in events:
        if e.get('pid') != anchor:
            continue
        if e.get('ph') == 'X':
            known.add(e.get('tid'))
        elif e.get('ph') == 'M' and e.get('name') in ('thread_name', 'thread_sort_index'):
            known.add(e.get('tid'))
            if e.get('name') == 'thread_sort_index':
                existing[e.get('tid')] = e.get('args', {}).get('sort_index', 0)
    ordered = sorted(known, key=lambda t: (existing.get(t, 0), t))
    at = min((ordered.index(t) for t in lead_tids if t in ordered), default=0)
    ordered[at:at] = [t for t in cpu_tids if t not in ordered]
    return [dict(ph='M', name='thread_sort_index', pid=anchor, tid=t, args=dict(sort_index=i))
            for i, t in enumerate(ordered)]


def split_steps(placed):
    """Split CPU spans into decode steps by layer-index wrap.

    A step is a monotonic run over the model's layers; the first span whose layer
    index is not greater than the previous one starts the next step. This is a
    heuristic: it is meaningless for a capture that does not repeat the layer
    stack, so callers sanity-check the result before trusting it.
    """
    steps, cur = [], []
    for s in placed:
        if cur and s['layer'] <= cur[-1]['layer']:
            steps.append(cur)
            cur = []
        cur.append(s)
    if cur:
        steps.append(cur)
    return steps


def step_stats(placed, intervals, ends):
    """Per-step masking numbers, plus the span list behind each step."""
    steps = split_steps(placed)
    if len(steps) < 3 or statistics.median(len(s) for s in steps) < 2:
        return [], []  # fewer than two wraps, or not a repeating layer stack
    t0 = min(x['ts'] for x in placed)
    rows = []
    for i, s in enumerate(steps):
        a, b = min(x['ts'] for x in s), max(x['cb'] for x in s)
        cpu = sum(x['cb'] - x['ca'] for x in s)
        busy = overlap(a, b, intervals, ends) if intervals else None
        ov = sum(x['ov'] for x in s) if intervals else None
        tokens = {x['tokens'] for x in s}
        span_ms = (b - a) / 1000
        single = len(tokens) == 1 and next(iter(tokens)) == 1
        rows.append(dict(
            step_index=i, start_ms=round((a - t0) / 1000, 3), span_ms=round(span_ms, 3),
            layers=len({x['layer'] for x in s}), samples=len(s),
            experts_sum=sum(x['expert_count'] for x in s),
            cpu_ms=round(cpu / 1000, 3),
            npu_busy_ms=None if busy is None else round(busy / 1000, 3),
            overlap_ms=None if ov is None else round(ov / 1000, 3),
            exposed_ms=None if ov is None else round((cpu - ov) / 1000, 3),
            masked_pct=None if not ov else round(100 * ov / cpu, 1),
            tokens=None if len(tokens) != 1 else next(iter(tokens)),
            tok_per_s=round(1000 / span_ms, 2) if single and span_ms > 0 else None))
    return rows, steps


def strip_lines(a, b, intervals, ends, spans, width=110):
    """Two rows over one step: NPU compute density, and how covered the CPU time is.

    The CPU row is graded by the share of that bin's CPU time that also had the
    NPU computing. Marking a bin as masked merely because one stray device op
    landed in it would read as near-perfect overlap.
    """
    if b <= a or not intervals:
        return []
    cpu_iv = union((s['ca'], s['cb']) for s in spans)
    cpu_ends = [y for _, y in cpu_iv]
    both = intersect(cpu_iv, intervals)
    both_ends = [y for _, y in both]
    bin_us = (b - a) / width
    npu_row, cpu_row = [], []
    for i in range(width):
        t0, t1 = a + i * bin_us, a + (i + 1) * bin_us
        frac = overlap(t0, t1, intervals, ends) / bin_us
        npu_row.append(' ' if frac <= .02 else ('.' if frac < .35 else ('+' if frac < .75 else '#')))
        cb = overlap(t0, t1, cpu_iv, cpu_ends)
        if cb <= 0:
            cpu_row.append(' ')
        else:
            masked = overlap(t0, t1, both, both_ends) / cb
            cpu_row.append('#' if masked >= .5 else ('+' if masked >= .15 else '-'))
    return [''.join(npu_row), ''.join(cpu_row)]


def roundtrip_windows(events):
    """Validate 3 input copies and either one combined or two split callbacks.

    No host enqueue timestamps and no nearest-copy fallback. Unknown patterns
    remain unmatched (including deferred experts with a fourth input copy).
    """
    pids = {e['pid'] for e in events if e.get('name') == 'process_name'
            and e.get('args', {}).get('name') == 'Ascend Hardware'}
    tracks = collections.defaultdict(list)
    for e in events:
        if e.get('pid') in pids and e.get('ph') == 'X':
            tracks[(e['pid'], e.get('tid'))].append(e)
    out = []
    for key, xs in tracks.items():
        xs.sort(key=lambda e: float(e['ts']))
        callbacks = [i for i, e in enumerate(xs) if e.get('name') == 'HOSTFUNC_CALLBACK']
        for n, i in enumerate(callbacks):
            if i+2 >= len(xs) or xs[i+1].get('name') != 'NOTIFY_WAIT':
                continue
            wait = xs[i+1]
            if xs[i+2].get('name') == 'MEMCPY_ASYNC':
                # Combined submit/drain: output H2D and output-ready event
                # follow this callback. Require the event to distinguish an
                # incomplete output from another layer's input copies.
                if i+3 >= len(xs) or xs[i+3].get('name') != 'DAVID_EVENT_RECORD':
                    continue
                join, result, callback_count = wait, xs[i+2], 1
            else:
                if n+1 >= len(callbacks):
                    continue
                j = callbacks[n+1]
                if j+2 >= len(xs):
                    continue
                join, result = xs[j+1], xs[j+2]
                if join.get('name') != 'NOTIFY_WAIT' or result.get('name') != 'MEMCPY_ASYNC':
                    continue
                callback_count = 2
            prev = callbacks[n-1] if n else -1
            inputs = [e for e in xs[prev+1:i] if e.get('name') == 'MEMCPY_ASYNC'
                      and e.get('args', {}).get('Task Type') == 'UBDMA']
            # Prior callback's output is also in this region; take only the
            # final three copies, and require the last two to be adjacent.
            if prev >= 0 and prev+2 < len(xs) and xs[prev+2].get('name') == 'MEMCPY_ASYNC':
                inputs = [e for e in inputs if e is not xs[prev+2]]
            if len(inputs) != 3 or xs[i-1] is not inputs[-1] or xs[i-2] is not inputs[-2]:
                continue
            inputs = inputs[-3:]
            a = float(inputs[0]['ts'])
            b = float(result['ts']) + float(result.get('dur', 0))
            out.append(dict(start=a, end=b, submit_start=float(wait['ts']),
                submit_end=float(wait['ts'])+float(wait.get('dur',0)),
                join_end=float(join['ts'])+float(join.get('dur',0)),
                callback_count=callback_count,
                pid=key[0], tid=key[1], input_task_ids=[e.get('args',{}).get('Task Id') for e in inputs],
                output_task_id=result.get('args',{}).get('Task Id')))
    return out


def merge(trace, records, include_actual_compute=False, actual_compute_only=False):
    if actual_compute_only and not include_actual_compute:
        raise ValueError('actual_compute_only requires include_actual_compute')
    events = trace if isinstance(trace, list) else trace['traceEvents']
    stamps = [(float(e['ts']), float(e['ts']) + float(e.get('dur', 0))) for e in events if 'ts' in e]
    if not stamps:
        raise ValueError('Ascend trace has no timestamped events')
    low, high = min(a for a, _ in stamps), max(b for _, b in stamps)
    analysis_pids = {e['pid'] for e in events if e.get('ph') == 'M' and e.get('name') == 'process_name'
                     and e.get('args', {}).get('name') == 'Overlap Analysis'}
    compute = [e for e in events if e.get('ph') == 'X' and e.get('name') == 'Computing' and e.get('pid') in analysis_pids]
    if not compute:
        compute = [e for e in events if e.get('ph') == 'X' and e.get('args', {}).get('Task Type') in
                   {'AI_CORE', 'AI_VECTOR_CORE', 'MIX_AIC', 'MIX_AIV', 'AIV', 'AIC'}]
    intervals = union((float(e['ts']), float(e['ts']) + float(e.get('dur', 0))) for e in compute)
    ends = [b for _, b in intervals]
    windows = roundtrip_windows(events)
    flushes, placed = {}, []
    actual = []
    unmatched = 0
    clipped = truncated = 0
    for r in records:
        if r.get('event') == 'cpu_moe.trace_flush':
            flushes[r['pid']] = r
            continue
        if r.get('event') != 'cpu_moe.compute':
            continue
        a = int(r['wall_ns']) / 1000
        dur = int(r['duration_ns']) / 1000
        b = a + dur
        if b <= low or a >= high or dur < 0:
            continue
        if include_actual_compute:
            actual.append((r, max(a, low), min(b, high)))
        # Submission only queues asynchronous work. The CPU worker may start
        # after the submit callback has returned, including during the join.
        # Require containment before this pair's join completion, and retain
        # the unique-match check instead of guessing a nearest callback.
        matches = [w for w in windows if w['submit_start'] <= a
                   and b <= w['join_end'] and b >= w['submit_start']]
        if len(matches) != 1:
            unmatched += 1
            continue
        boundary = matches[0]
        compute_a, compute_b, compute_dur = a, b, dur
        a, b = boundary['start'], boundary['end']
        dur = b - a
        ca, cb = max(a, low), min(b, high)
        clipped += int(ca != a or cb != b)
        truncated += bool(r.get('expert_ids_truncated'))
        placed.append(dict(layer=int(r['layer']), part=int(r['part']), ts=a, dur=dur, ca=ca, cb=cb,
                           ov=overlap(ca, cb, intervals, ends) if intervals else None,
                           expert_count=r['expert_count'], tokens=r.get('tokens'), record=r, boundary=boundary, compute_us=compute_dur, compute_start=compute_a, compute_end=compute_b))
    placed.sort(key=lambda p: (p['ts'], p['part']))

    anchor, lead_tid, anchor_name = pick_anchor(events)
    parts = sorted({p['part'] for p in placed})
    tid_for = {}
    if anchor is not None:
        tid_for = reserve_tids(events, anchor, lead_tid, parts)
        for p in placed:
            p['track_pid'], p['track_tid'] = anchor, tid_for[p['part']]
    else:  # nothing to sit next to; keep the detached process
        for p in placed:
            p['track_pid'], p['track_tid'] = 2000000000 + int(p['record']['pid']), p['part']

    output, groups = [], collections.defaultdict(list)
    for p in placed:
        r, dur, ca, cb, ov = p['record'], p['dur'], p['ca'], p['cb'], p['ov']
        args = dict(r)
        args.update(layer_index=p['layer'], cpu_experts=r['expert_count'], cpu_compute_us=p['compute_us'], roundtrip_us=dur,
                    compute_start_us=p['compute_start'], compute_end_us=p['compute_end'],
                    device_boundary=p['boundary'],
                    coordinator_cpu_us=r.get('coordinator_cpu_ns', 0) / 1000,
                    npu_compute_overlap_us=ov,
                    npu_compute_overlap_pct=100 * ov / (cb - ca) if ov is not None and cb > ca else None,
                    clipped_to_profile_window=(ca != p['ts'] or cb != p['ts'] + dur),
                    note='Actual device input-copy start through output-copy completion; CPU compute is an inner interval')
        output.append(dict(name=cpu_span_name(r), cat='cpu_moe_roundtrip', ph='X',
                           pid=p['track_pid'], tid=p['track_tid'], ts=ca, dur=cb - ca, args=args))
        groups[p['layer']].append((dur, r['expert_count'], cb - ca, ov))

    if anchor is None:
        for tpid, ttid, part in sorted({(p['track_pid'], p['track_tid'], p['part']) for p in placed}):
            output.extend([
                dict(ph='M', name='process_name', pid=tpid, tid=0, args=dict(name='CPU expert computation')),
                dict(ph='M', name='thread_name', pid=tpid, tid=ttid, args=dict(name=f'CPU MoE / part {part}')),
                dict(ph='M', name='thread_sort_index', pid=tpid, tid=ttid, args=dict(sort_index=ttid))])
    else:
        for part in parts:
            output.append(dict(ph='M', name='thread_name', pid=anchor, tid=tid_for[part],
                               args=dict(name=f'CPU MoE / part {part}')))
        output.extend(reorder_threads(events, anchor, [tid_for[p] for p in parts], [lead_tid]))

    per_layer = {}
    for layer, rows in sorted(groups.items()):
        durations = sorted(r[0] for r in rows)
        total, ov = sum(r[2] for r in rows), sum(r[3] or 0 for r in rows)
        per_layer[str(layer)] = dict(layer_label=f'L{layer+1}', samples=len(rows),
            mean_roundtrip_us=statistics.mean(durations), p50_roundtrip_us=statistics.median(durations),
            mean_compute_us=statistics.mean(p['compute_us'] for p in placed if p['layer']==layer), p50_compute_us=statistics.median(p['compute_us'] for p in placed if p['layer']==layer),
            p90_roundtrip_us=durations[min(len(durations)-1, int(.9*len(durations)))],
            mean_cpu_experts=statistics.mean(r[1] for r in rows),
            npu_compute_overlap_pct=100*ov/total if intervals and total else None)
    # A sparse matched subset cannot represent complete decode steps.
    steps, step_spans = (step_stats(placed, intervals, ends)
                        if unmatched <= .1 * (len(placed) + unmatched) else ([], []))
    summary = dict(cpu_spans=sum(len(v) for v in groups.values()), clipped_spans=clipped, unmatched_compute_spans=unmatched, interval_kind='npu_cpu_roundtrip',
        truncated_expert_lists=truncated, layers=per_layer, steps=steps,
        cpu_track=dict(pid=anchor, host=anchor_name, lead_tid=lead_tid, kind='anchored')
        if anchor is not None else dict(pids=sorted({p['track_pid'] for p in placed}), kind='detached'),
        writer_status=list(flushes.values()), npu_overlap_available=bool(intervals),
        overlap_definition='Intersection of each NPU-CPU roundtrip interval with union of all NPU compute intervals; not hidden latency or guaranteed speedup')
    if include_actual_compute:
        # Default viewing can omit sparse roundtrip bars; validated boundary
        # details remain available on the corresponding actual CPU events.
        boundaries = {id(p['record']): p['boundary'] for p in placed}
        if actual_compute_only:
            output = []
        actual_parts = sorted({int(r['part']) for r, _, _ in actual})
        actual_tids = (reserve_tids(events + output, anchor, lead_tid, actual_parts)
                       if anchor is not None else {})
        for r, a, b in actual:
            part = int(r['part'])
            pid = anchor if anchor is not None else 2100000000 + int(r['pid'])
            tid = actual_tids[part] if anchor is not None else part
            args = dict(r)
            if id(r) in boundaries:
                args['device_boundary'] = boundaries[id(r)]
            args.update(cpu_compute_us=int(r['duration_ns']) / 1000,
                        clipped_to_profile_window=(a != int(r['wall_ns'])/1000
                            or b != (int(r['wall_ns'])+int(r['duration_ns']))/1000),
                        note='Actual CPU CLOCK_REALTIME start and CLOCK_MONOTONIC duration; '
                             'not shifted to device callback timestamps')
            output.append(dict(ph='X', name=cpu_span_name(r),
                               cat='cpu_moe_actual_compute', pid=pid, tid=tid,
                               ts=a, dur=b-a, args=args))
        if anchor is not None:
            for part in actual_parts:
                output.append(dict(ph='M',name='thread_name',pid=anchor,tid=actual_tids[part],
                                   args=dict(name=f'CPU MoE actual compute / part {part}')))
            output.extend(reorder_threads(events + output, anchor,
                [actual_tids[p] for p in actual_parts],
                ([lead_tid] if actual_compute_only else
                 [tid_for[p] for p in parts] or [lead_tid])))
        else:
            for pid, part in sorted({(2100000000+int(r['pid']),int(r['part'])) for r,_,_ in actual}):
                output.extend([dict(ph='M',name='process_name',pid=pid,args=dict(name='CPU expert actual compute')),
                               dict(ph='M',name='thread_name',pid=pid,tid=part,args=dict(name=f'CPU MoE actual compute / part {part}'))])
        summary['actual_compute_spans'] = len(actual)
        summary['roundtrip_track_included'] = not actual_compute_only
        summary['actual_compute_time_basis'] = 'CPU realtime start plus monotonic duration, without device clock alignment'
    if steps:
        # Zoom on a few layer cycles rather than the whole step: at one bin per
        # step the CPU spans alias into noise instead of showing the alternation.
        mid = sorted(range(len(steps)), key=lambda i: steps[i]['span_ms'])[len(steps) // 2]
        zoom = step_spans[mid][:8] or step_spans[mid]
        a, b = min(x['ts'] for x in zoom), max(x['cb'] for x in zoom)
        lines = strip_lines(a, b, intervals, ends, zoom)
        if lines:
            summary['strip'] = dict(step_index=steps[mid]['step_index'],
                                    start_ms=steps[mid]['start_ms'], span_ms=(b - a) / 1000,
                                    layer_span=f"L{zoom[0]['layer']+1}-L{zoom[-1]['layer']+1}",
                                    lines=lines)
    result = list(events) + output if isinstance(trace, list) else dict(trace, traceEvents=list(events)+output)
    return result, summary


def render_report(summary):
    def num(v, w=7, p=2):
        return f"{'-':>{w}}" if v is None else f"{v:>{w}.{p}f}"
    out = ['CPU MoE roundtrip vs NPU compute', '=' * 88]
    track = summary['cpu_track']
    out.append(f"CPU spans        : {summary['cpu_spans']} "
               f"(clipped {summary['clipped_spans']}, truncated expert lists {summary['truncated_expert_lists']})")
    out.append(f"unmatched        : {summary['unmatched_compute_spans']} CPU records lack a unique validated roundtrip")
    if 'actual_compute_spans' in summary:
        out.append(f"actual CPU spans : {summary['actual_compute_spans']} on a separate raw CPU clock track")
    out.append(f"CPU track        : " + (
        f"in {track['host']} (pid {track['pid']}), directly above device track {track['lead_tid']}"
        if track['kind'] == 'anchored'
        else f"detached, pids {track['pids']} (no Ascend Hardware process found)"))
    for w in summary['writer_status']:
        out.append(f"writer           : written={w.get('written')} dropped={w.get('dropped')}")
    out.append(f"overlap basis    : {summary['overlap_definition']}")
    out.append('')
    if not summary['steps']:
        out.append('No complete decode-step table: insufficient matched coverage or repeating layers.')
        out.append('Per-layer totals are still in the statistics JSON.')
        return '\n'.join(out) + '\n'
    out.append('Per decode step (step = one pass over the layer stack, split on layer-index wrap)')
    out.append('')
    out.append(f"{'#':>3} {'start_ms':>9} {'span_ms':>8} {'layers':>7} {'experts':>8} {'cpu_ms':>7} "
               f"{'npu_ms':>7} {'ovl_ms':>7} {'exposed':>8} {'masked%':>8} {'tok/s':>7}")
    out.append('-' * 88)
    for s in summary['steps']:
        out.append(f"{s['step_index']:>3} {s['start_ms']:>9.2f} {s['span_ms']:>8.2f} {s['layers']:>7} "
                   f"{s['experts_sum']:>8} {num(s['cpu_ms'])} {num(s['npu_busy_ms'])} {num(s['overlap_ms'])} "
                   f"{num(s['exposed_ms'], 8)} {num(s['masked_pct'], 8, 1)} {num(s['tok_per_s'])}")
    steady = [s for s in summary['steps'] if s['masked_pct'] is not None][1:]
    if steady:
        out.append('')
        out.append(f"steady steps     : {len(steady)} (first step dropped as warmup)")
        for key, label, unit in (('span_ms', 'step', 'ms'), ('cpu_ms', 'CPU MoE', 'ms'),
                                 ('npu_busy_ms', 'NPU compute', 'ms'), ('exposed_ms', 'CPU exposed', 'ms'),
                                 ('masked_pct', 'masked', '%')):
            vals = [s[key] for s in steady if s[key] is not None]
            if vals:
                out.append(f"  mean {label:<12}= {statistics.mean(vals):>8.2f} {unit}")
    if summary.get('strip'):
        st = summary['strip']
        out.append('')
        out.append(f"Layer-cycle detail (step {st['step_index']}, layers {st['layer_span']}, "
                   f"{st['span_ms']:.2f} ms, {len(st['lines'][0])} bins)")
        out.append('')
        out.append('  NPU compute |' + st['lines'][0])
        out.append('  CPU MoE     |' + st['lines'][1])
        out.append('')
        out.append("  legend  NPU row: ' .+# ' none .. all of the bin spent computing")
        out.append("          CPU row: '-' <15% of the bin's CPU time overlapped by NPU compute,")
        out.append("                   '+' 15-50%, '#' >=50% (well masked)")
    out.append('')
    out.append('Per layer (all steps pooled)')
    out.append('')
    out.append(f"{'layer':>7} {'n':>4} {'mean_exp':>9} {'mean_us':>9} {'p50_us':>9} {'p90_us':>9} {'ovl%':>7}")
    out.append('-' * 60)
    for layer in sorted(summary['layers'], key=int):
        v = summary['layers'][layer]
        out.append(f"{v['layer_label']:>7} {v['samples']:>4} {v['mean_cpu_experts']:>9.2f} "
                   f"{v['mean_roundtrip_us']:>9.1f} {v['p50_roundtrip_us']:>9.1f} {v['p90_roundtrip_us']:>9.1f} "
                   f"{num(v['npu_compute_overlap_pct'], 7, 1)}")
    return '\n'.join(out) + '\n'


def read_records(paths):
    for path in paths:
        with path.open(encoding='utf8') as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # an unfinished final line from a live writer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ascend_trace', type=Path)
    parser.add_argument('cpu_trace', type=Path, nargs='+')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--stats', type=Path)
    parser.add_argument('--report', type=Path, help='human-readable masking report')
    parser.add_argument('--include-actual-compute', action='store_true',
                        help='also retain all in-window raw CPU spans on a separate track')
    parser.add_argument('--actual-compute-only', action='store_true',
                        help='display just the actual CPU track; retain matched boundary arguments')
    args = parser.parse_args()
    protected = {args.ascend_trace.resolve(), *(p.resolve() for p in args.cpu_trace)}
    if args.out.resolve() in protected:
        parser.error('output must differ from all inputs')
    for extra in (args.stats, args.report):
        if extra and extra.resolve() in protected | {args.out.resolve()}:
            parser.error(f'{extra.name} must differ from all inputs and the merged trace')
    trace = json.loads(args.ascend_trace.read_text(encoding='utf8'))
    if args.actual_compute_only and not args.include_actual_compute:
        parser.error('--actual-compute-only requires --include-actual-compute')
    merged, summary = merge(trace, read_records(args.cpu_trace), args.include_actual_compute,
                            args.actual_compute_only)
    if not summary['cpu_spans'] and not summary.get('actual_compute_spans'):
        parser.exit(2, 'ERROR: zero CPU roundtrip spans matched; no merged output written. unmatched_compute_spans=' + str(summary['unmatched_compute_spans']) + '\n')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, separators=(',', ':')), encoding='utf8')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(summary), encoding='utf8')
    summary.pop('strip', None)
    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(json.dumps(summary, indent=2), encoding='utf8')
    written = [f"cpu spans   {summary['cpu_spans']} -> {args.out}"]
    if 'actual_compute_spans' in summary:
        written.append(f"actual CPU  {summary['actual_compute_spans']} (raw CPU clock track)")
    for label, path in (('report', args.report), ('stats', args.stats)):
        if path:
            written.append(f"{label:<11} {path}")
    print('\n'.join(written))
    if not args.stats:
        print(json.dumps(summary, indent=2))  # no file holds it, so show it here
    if not summary['cpu_spans']:
        print('ERROR: no CPU roundtrip spans matched. Check unmatched_compute_spans and hardware communication pattern.')
    if any(r.get('dropped', 0) for r in summary['writer_status']):
        print('WARNING: bounded CPU trace queue dropped samples; inspect writer_status.')
    if not summary['steps']:
        print('No decode steps detected; per-step masking table omitted.')


if __name__ == '__main__':
    main()
