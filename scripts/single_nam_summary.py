import json,numpy as np
from single_nam_common import OUT as W
runs={"10":"ten_capture","5":"cap5","3":"three_capture","2":"cap2"}
gs=sorted([str(float(g)) for g in range(1,11)]+[str(g+.5) for g in range(1,10)],key=float)
ev={k:json.load(open(W/f"eval_{v}.json")) for k,v in runs.items()}
print("raw ESR, mean of 3 held-out DIs | 10cap 5cap 3cap 2cap | G5-same-law   (G8.5 excluded)")
tot={k:[] for k in list(runs)+["g5"]}
for g in gs:
    if g=="8.5": continue
    row=[]
    for k in runs:
        v=np.mean([ev[k]["per_di"][d][g]["new"]["raw_esr"] for d in ev[k]["per_di"]]); row.append(v); tot[k].append(v)
    b=np.mean([ev["10"]["per_di"][d][g]["g5_same_law"]["raw_esr"] for d in ev["10"]["per_di"]]); tot["g5"].append(b)
    mark=["*" if ev[k]["per_di"]["moderate_brit"][g]["trained"] else " " for k in runs]
    print(f"G{float(g):4.1f} | "+" ".join(f"{x:.4f}{m}" for x,m in zip(row,mark))+f" | {b:.4f}")
print("mean ",{k:round(float(np.mean(v)),4) for k,v in tot.items()})
print("worst",{k:round(float(np.max(v)),4) for k,v in tot.items()})
