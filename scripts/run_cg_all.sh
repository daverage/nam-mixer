#!/bin/bash
# Build + train 10/5/3-capture Continuous Gain v3 models for each amp (sequential). Resumable: skips existing bundles/models.
cd "$(dirname "$0")"
for amp in "$@"; do
  D=$(cd .. && pwd)/work/cg/$amp; mkdir -p $D
  export SINGLE_NAM_AMP=$amp
  [ -f $D/cg_10/manifest.json ] || ../.venv/bin/python cg_build.py --out $D/cg_10 > $D/build_10.log 2>&1
  [ -f $D/cg_5/manifest.json ]  || ../.venv/bin/python cg_build.py --gains 1 3 5 7 10 --out $D/cg_5 > $D/build_5.log 2>&1
  [ -f $D/cg_3/manifest.json ]  || ../.venv/bin/python cg_build.py --gains 1 5 10 --out $D/cg_3 > $D/build_3.log 2>&1
  for n in 10 3 5; do
    ls $D/cg_$n/*/*.nam >/dev/null 2>&1 || ../.venv-a2/bin/python single_nam_train.py $D/cg_$n --epochs 60 --name ${amp}_ContinuousGain_${n}Captures > $D/train_$n.log 2>&1
  done
done
