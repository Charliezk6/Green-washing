from __future__ import annotations
import asyncio, csv, json, os, random, re
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'deepgreen_reproduction/04_layer_b_rag/results/unique_task_results.jsonl'
OUT=ROOT/'deepgreen_reproduction/04_layer_b_rag/validation'
MODEL='qwen3.6-35b-a3b-fp8'; BASE='https://token.nau.edu.cn/v1'
PROMPT='''你是独立的ESG披露人工复核员。不要参考其他模型的标签。根据目标词及完整年报句子，判断公司是否披露了围绕目标词的实际、具体环保行动。具体建设、改造、安装、运行、处理、回收、净化、减排或实施企业自身方案为1；仅栏目标题、愿景口号、重视环保、遵守/引用法规标准、取得许可、列示污染物或排放数据、主体或行动不明确为0。只返回纯JSON：{"label":0或1,"reason":"一句简短理由"}。'''

def parse(s):
 m=re.search(r'\{[\s\S]*?\}',s); p=json.loads(m.group()); v=p['label']
 if isinstance(v,bool) or str(v) not in {'0','1'}: raise ValueError('label')
 return int(v),str(p.get('reason',''))
async def one(c,sem,key,row):
 error=''
 for attempt in range(1,5):
  try:
   async with sem:
    x=await c.post(BASE+'/chat/completions',headers={'Authorization':'Bearer '+key},json={'model':MODEL,'messages':[{'role':'system','content':PROMPT},{'role':'user','content':f"目标词：{row['word']}\n完整句子：{row['marked_sentence']}"}],'temperature':0,'max_tokens':128,'stream':False,'chat_template_kwargs':{'enable_thinking':False}})
   x.raise_for_status(); content=x.json()['choices'][0]['message']['content']; label,reason=parse(content); return {**row,'audit_label':label,'audit_reason':reason}
  except Exception as exc:
   error=f'{type(exc).__name__}: {exc}'
   if attempt<4: await asyncio.sleep(2**(attempt-1)+random.random())
 raise RuntimeError(error)
async def main():
 rows=[json.loads(x) for x in SOURCE.open(encoding='utf-8') if x.strip()]
 sample_size=1000; per_class=sample_size//2
 random.seed(20260717); groups={0:[x for x in rows if x['judgment']==0],1:[x for x in rows if x['judgment']==1]}; sample=random.sample(groups[0],per_class)+random.sample(groups[1],per_class); random.shuffle(sample)
 key=os.environ['NAU_API_KEY']; sem=asyncio.Semaphore(100)
 async with httpx.AsyncClient(timeout=httpx.Timeout(120,connect=30),limits=httpx.Limits(max_connections=100)) as c: out=await asyncio.gather(*(one(c,sem,key,r) for r in sample))
 OUT.mkdir(parents=True,exist_ok=True); fields=['word','sentence','marked_sentence','judgment','confidence','audit_label','audit_reason'];
 with (OUT/'simulated_manual_audit_1000.csv').open('w',encoding='utf-8-sig',newline='') as h: w=csv.DictWriter(h,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(out)
 tp=sum(x['judgment']==1 and x['audit_label']==1 for x in out); tn=sum(x['judgment']==0 and x['audit_label']==0 for x in out); fp=sum(x['judgment']==1 and x['audit_label']==0 for x in out); fn=sum(x['judgment']==0 and x['audit_label']==1 for x in out); acc=(tp+tn)/sample_size; prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0; f1=2*prec*rec/(prec+rec) if prec+rec else 0; den=((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5; mcc=(tp*tn-fp*fn)/den if den else 0
 summary={'sample':sample_size,'sampling':'500 model-negative + 500 model-positive','tp':tp,'tn':tn,'fp':fp,'fn':fn,'accuracy':acc,'precision':prec,'recall':rec,'f1':f1,'mcc':mcc,'disagreements':fp+fn,'note':'AI-simulated independent audit; not human annotation'}; (OUT/'simulated_manual_audit_1000_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False))
asyncio.run(main())
