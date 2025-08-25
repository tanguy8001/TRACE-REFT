# point to your predictions folder
#P="/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/O-LoRA-500samples-llama27Bchat/predictions"
P="/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/REFT-CL/predictions"
print(f"Reading from {P}")

import json, glob, os, re
task_metric = {
  "C-STANCE":"accuracy","FOMC":"accuracy","ScienceQA":"accuracy",
  "NumGLUE-cm":"accuracy","NumGLUE-ds":"accuracy",
  "MeetingBank":"rouge-L","Py150":"similarity","20Minuten":"sari"
}
def norm(task, metric, v): return v/100.0 if metric in ("similarity","sari") else v

rows={}
for f in sorted(glob.glob(os.path.join(P,"results-*-*-*.json"))):
    m=re.match(r".*results-(\d+)-\d+-(.+)\.json$", f); 
    if not m: continue
    r=int(m.group(1)); task=m.group(2)
    d=json.load(open(f)); metric=task_metric.get(task)
    if metric is None: continue
    v=d.get("eval",{}).get(metric)
    if isinstance(v,(int,float)): rows.setdefault(task,{})[r]=(metric,v)

rounds=sorted({r for m in rows.values() for r in m})
print("rounds (0-based):", rounds)
order=["C-STANCE","FOMC","MeetingBank","Py150","ScienceQA","NumGLUE-cm","NumGLUE-ds","20Minuten"]
for t in order:
    print(t, [None if r not in rows.get(t,{}) else round(norm(t, *rows[t][r]),3) for r in rounds])
final=rounds[-1]
vals=[norm(t,*rows[t][final]) for t in order if final in rows.get(t,{})]
print("final-round average:", round(sum(vals)/len(vals),3))