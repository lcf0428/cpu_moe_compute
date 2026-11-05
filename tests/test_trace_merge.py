import importlib.util
from decimal import Decimal
from pathlib import Path
spec=importlib.util.spec_from_file_location('merge_trace',Path(__file__).resolve().parent.parent/'scripts'/'merge_cpu_compute_trace.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
record=dict(event='cpu_moe.compute',pid=42,layer=3,part=0,expert_count=1,expert_route_count=1,transfer_count=2,mono_raw_ns=1500000,wall_ns=999999999999999999,duration_ns=100000)
events=[dict(ph='X',name='CPU_MOE_CLOCK_ANCHOR_a',ts=1000,dur=1),dict(ph='X',name='CPU_MOE_CLOCK_ANCHOR_b',ts=2000,dur=1),dict(ph='X',name='kept-device-op',ts='1510.125',dur='0.125',pid=1,tid=1)]
anchors=[dict(marker_name='CPU_MOE_CLOCK_ANCHOR_a',pid=42,before_raw_ns=999990,after_enter_raw_ns=1000010),dict(marker_name='CPU_MOE_CLOCK_ANCHOR_b',pid=42,before_raw_ns=1999990,after_enter_raw_ns=2000010)]
result,stats=m.merge(events,[record],True,True,anchors)
actual=[e for e in result if e.get('cat')=='cpu_moe_actual_compute'][0]
assert actual['name']=='L4_E1_C1_T2'
assert Decimal(actual['ts'])==1500 and Decimal(actual['dur'])==100
assert stats['export_window']==dict(start_us='1500.000',end_us='1600.000')
assert next(e for e in result if e.get('name')=='kept-device-op')['dur']=='0.125'
assert m.cpu_span_name(dict(record,transfer_count=0)).endswith('_T0')
legacy=dict(record);legacy.pop('transfer_count');assert m.cpu_span_name(legacy).endswith('_Tunknown')
for bad in [None,anchors[:1]]:
    try: m.merge(events,[record],True,True,bad)
    except ValueError: pass
    else: raise AssertionError('missing calibration accepted')
large='1800000000000000'
trace=[dict(ph='X',name='inside',ts=large+'.375',dur='0.125'),dict(ph='X',name='crossing',ts=large+'.125',dur='0.375'),dict(ph='M',name='thread_name',args={'name':'original'})]
clipped,window=m.crop_to_cpu_window(trace,[(None,Decimal(large+'.25'),Decimal(large+'.625'))])
assert clipped[0]==trace[0]
assert clipped[1]['ts']==large+'.25' and clipped[1]['dur']=='0.250'
assert clipped[2]==trace[2]
print('PASS: actual T counts/unknown, clock-offset mapping, missing-anchor refusal, exact large-timestamp clipping and original-event preservation')

original_windows=m.roundtrip_windows
m.roundtrip_windows=lambda events:[dict(start=1400.,end=1700.,submit_start=1450.,join_end=1650.)]
result,stats=m.merge(events,[record,dict(record,mono_raw_ns=500000)],True,True,anchors)
assert stats['export_window']==dict(start_us='1400.0',end_us='1700.0')
assert stats['cpu_records_outside_clock_window']==1
m.roundtrip_windows=original_windows
print('PASS: export window retains complete matched roundtrip plus CPU, excludes pre-capture CPU records')

result,stats=m.merge(events,[dict(record,event='cpu_moe.compute_total',part=-1),record],True,True,anchors,compute_event='cpu_moe.compute_total')
actual=[e for e in result if e.get('cat')=='cpu_moe_actual_compute']
assert len(actual)==1 and actual[0]['args']['event']=='cpu_moe.compute_total'
assert stats['compute_event']=='cpu_moe.compute_total'
print('PASS: explicit total event selection excludes nested part event')
