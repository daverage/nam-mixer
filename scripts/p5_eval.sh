#!/bin/bash
# Expansion: whole evaluation chain for one amp (run after training). Usage: p5_eval.sh <amp>
cd "$(dirname "$0")"; amp=$1; PY=../.venv-a2/bin/python; L=../work/p4e/final/logs
$PY fc_eval.py $amp FC_s0 >> $L/eval_$amp.log 2>&1; $PY fc_eval.py $amp BASE >> $L/eval_$amp.log 2>&1
$PY fc_sweep.py $amp FC_s0 >> $L/eval_$amp.log 2>&1; $PY fc_sweep.py $amp BASE >> $L/eval_$amp.log 2>&1
$PY p5_extremes.py $amp >> $L/eval_$amp.log 2>&1; $PY p5_verify.py $amp >> $L/eval_$amp.log 2>&1
for p in $($PY -c "import json;print(*[('%g'%g) for g in json.load(open('../work/p4e/$amp/real_ref.json'))['gains']])"); do $PY fc_cases.py $amp $p >> $L/eval_$amp.log 2>&1; done
echo done > $L/eval_done_$amp.txt
