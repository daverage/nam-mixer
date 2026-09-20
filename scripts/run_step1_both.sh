A=$1; O=../work/single_nam_$A
SINGLE_NAM_AMP=$A ../.venv-a2/bin/python single_nam_step1_analysis.py > /private/tmp/claude-501/x/$A.esr.log 2>&1
cp $O/control_law.json $O/control_law_esr_search.json; cp $O/conflict_pairs.json $O/conflict_pairs_esr.json
SINGLE_NAM_AMP=$A SINGLE_NAM_LAW=level ../.venv-a2/bin/python single_nam_step1_analysis.py > /private/tmp/claude-501/x/$A.level.log 2>&1
SINGLE_NAM_AMP=$A ../.venv-a2/bin/python - > /private/tmp/claude-501/x/$A.rms.log 2>&1 <<'PY'
from single_nam_common import *
x=load_di("moderate_brit")[:SR*20]
print([round(float(rms_dbfs(render_capture(g,x))),1) for g in INT_GAINS])
PY
