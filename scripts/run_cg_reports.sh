#!/bin/bash
# Wait for each amp's 10/3/5-capture models, then run cg_report + cg_summary for it.
cd "$(dirname "$0")"
for amp in "$@"; do
  D=$(cd .. && pwd)/work/cg/$amp
  until ls $D/cg_5/*/*.nam >/dev/null 2>&1; do sleep 60; done
  SINGLE_NAM_AMP=$amp ../.venv/bin/python cg_report.py --amp $amp --root $D > $D/report.log 2>&1
  ../.venv/bin/python cg_summary.py $amp > $D/summary.md 2>> $D/report.log
done
