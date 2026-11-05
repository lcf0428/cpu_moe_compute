#!/usr/bin/env python3
"""Host-only monotonic phase analysis; device clocks are never subtracted."""
import json,sys,collections,statistics,pathlib
trace_path=pathlib.Path(sys.argv[1]);raw_path=pathlib.Path(sys.argv[2])
d=json.loads(trace_path.read_text());events=d if isinstance(d,list) else d['traceEvents']
ids={e.get('args',{}).get('task_id') for e in events if e.get('cat')=='cpu_moe_actual_compute'}-{None,0}
groups=collections.defaultdict(list);flush=[]
for line in raw_path.open():
 r=json.loads(line)
 if r.get('event')=='cpu_moe.lifecycle' and r['task_id'] in ids:groups[r['task_id']].append(r)
 if r.get('event')=='cpu_moe.trace_flush':flush.append(r)
pairs=[('callback_enter','queue_publish'),('queue_publish','queue_take'),('queue_publish','submit_return'),('submit_return','drain_enter'),('queue_take','compute_begin'),('compute_begin','compute_end'),('compute_end','task_return'),('task_return','completion_publish'),('completion_publish','completion_observed'),('completion_observed','drain_return'),('drain_return','callback_exit'),('callback_enter','callback_exit')]
summary={};lines=['CPU MoE host lifecycle: all durations use CLOCK_MONOTONIC','Device notify/D2H/H2D/join remain in Ascend trace; no cross-clock subtraction.',f'Profile tasks={len(ids)}, tasks with lifecycle={len(groups)}',f'Writer latest={flush[-1] if flush else None}','phase                                      n     median_us      p90_us      p99_us']
for a,b in pairs:
 vals=[]
 for tid,rs in groups.items():
  phases={x['phase']:x for x in rs if x['phase'] not in ('worker_begin','worker_end')}
  if a in phases and b in phases:
   v=(phases[b]['mono_ns']-phases[a]['mono_ns'])/1000
   if v>=0:vals.append(v)
 if vals:
  vals.sort(); q=lambda p:vals[min(len(vals)-1,int((len(vals)-1)*p))]
  summary[a+' -> '+b]={'n':len(vals),'median_us':statistics.median(vals),'p90_us':q(.9),'p99_us':q(.99)}
  lines.append(f'{a+" -> "+b:42} {len(vals):4d} {statistics.median(vals):12.2f} {q(.9):11.2f} {q(.99):11.2f}')
extra=[]
for rs in groups.values():
 ph={x['phase']:x for x in rs}
 if all(k in ph for k in ['callback_enter','callback_exit','compute_begin','compute_end']):
  value=(ph['callback_exit']['mono_ns']-ph['callback_enter']['mono_ns']-ph['compute_end']['mono_ns']+ph['compute_begin']['mono_ns'])/1000
  if value>=0:extra.append(value)
if extra:
 extra.sort();q=lambda p:extra[min(len(extra)-1,int((len(extra)-1)*p))]
 summary['host callback non-compute critical path']={'n':len(extra),'median_us':statistics.median(extra),'p90_us':q(.9),'p99_us':q(.99)}
 lines.append(f'{"host callback non-compute critical path":42} {len(extra):4d} {statistics.median(extra):12.2f} {q(.9):11.2f} {q(.99):11.2f}')
lines.append('Submission metadata can overlap CPU work; stage medians must not be summed.')
worker_metrics=collections.defaultdict(list)
for rs in groups.values():
 rs=sorted(rs,key=lambda x:x['mono_ns'])
 dispatch=[x for x in rs if x['phase']=='pool_dispatch']
 for i,x in enumerate(dispatch):
  end=dispatch[i+1]['mono_ns'] if i+1<len(dispatch) else 2**63
  window=[z for z in rs if x['mono_ns']<=z['mono_ns']<end]
  begins=[z['mono_ns'] for z in window if z['phase']=='worker_begin']
  finishes=[z['mono_ns'] for z in window if z['phase']=='worker_end']
  waits=[z['mono_ns'] for z in window if z['phase']=='pool_wait_return']
  if begins:
   worker_metrics['pool_dispatch -> first worker'].append((min(begins)-x['mono_ns'])/1000)
   worker_metrics['pool_dispatch -> last recorded worker'].append((max(begins)-x['mono_ns'])/1000)
  if finishes and waits and max(waits)>=max(finishes):
   worker_metrics['last recorded worker -> pool wait return'].append((max(waits)-max(finishes))/1000)
for label,vals in worker_metrics.items():
 vals.sort();q=lambda p:vals[min(len(vals)-1,int((len(vals)-1)*p))]
 summary[label]={'n':len(vals),'median_us':statistics.median(vals),'p90_us':q(.9),'p99_us':q(.99)}
 lines.append(f'{label:42} {len(vals):4d} {statistics.median(vals):12.2f} {q(.9):11.2f} {q(.99):11.2f}')
phase_cpus=collections.defaultdict(collections.Counter)
for rs in groups.values():
 for x in rs:phase_cpus[x['phase']][x['cpu']]+=1
lines.append('Most frequent CPU cores per phase:')
for name in ['callback_enter','queue_take','compute_begin','completion_observed','callback_exit']:
 lines.append(f'  {name}: {phase_cpus[name].most_common(5)}')
lines.append('Worker maxima use recorded workers only; dropped records can hide the true last worker.')
# Instant events retain task/thread/core linkage, on an explicitly raw host clock track.
pid=2200000000
out=list(events)+[dict(ph='M',name='process_name',pid=pid,args={'name':'CPU lifecycle / raw host realtime'})]
tids=set()
for rs in groups.values():
 for x in rs:
  tids.add(x['tid']);out.append(dict(ph='i',s='t',name=f"L{x['layer']+1} {x['phase']} T{x['task_id']}",pid=pid,tid=x['tid'],ts=x['wall_ns']/1000,args=x))
for task_id,rs in groups.items():
 phases={x['phase']:x for x in rs if x['phase'] not in ('worker_begin','worker_end')}
 for a,b in pairs:
  if a in phases and b in phases and phases[b]['mono_ns']>=phases[a]['mono_ns']:
   x,y=phases[a],phases[b]
   out.append(dict(ph='X',name=f"L{x['layer']+1} {a} -> {b} T{task_id}",cat='cpu_moe_host_stage',pid=pid,tid=x['tid'],ts=x['wall_ns']/1000,dur=(y['mono_ns']-x['mono_ns'])/1000,args={'task_id':task_id,'layer':x['layer'],'begin_cpu':x['cpu'],'end_cpu':y['cpu'],'time_basis':'host monotonic duration, raw realtime start'}))
for tid in tids:out.append(dict(ph='M',name='thread_name',pid=pid,tid=tid,args={'name':f'Host TID {tid}'}))
if isinstance(d,list):merged=out
else:d['traceEvents']=out;merged=d
base=trace_path.parent
(base/'trace_view_cpu_lifecycle.json').write_text(json.dumps(merged,separators=(',',':')))
(base/'cpu_lifecycle_stats.json').write_text(json.dumps({'profile_tasks':len(ids),'lifecycle_tasks':len(groups),'writer':flush[-1] if flush else None,'phases':summary},indent=2))
(base/'cpu_lifecycle_report.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
