import concurrent.futures,json,time,urllib.request
from pathlib import Path
root=Path('/mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/native_npu_integration')
questions=['What is 1+1? Reply with the number only.','What is the capital of China? Reply with the city name only.','请用一句话解释什么是CPU。']
def request(question):
    payload={'model':'deepseek_v4','messages':[{'role':'user','content':question}],'max_tokens':96,'temperature':0,'chat_template_kwargs':{'thinking':False,'enable_thinking':False}}
    req=urllib.request.Request('http://127.0.0.1:8150/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    start=time.perf_counter()
    with urllib.request.urlopen(req,timeout=180) as response: result=json.load(response)
    return {'question':question,'seconds':time.perf_counter()-start,'response':result}
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for result in pool.map(request,questions):
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
(root/'service_smoke_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
