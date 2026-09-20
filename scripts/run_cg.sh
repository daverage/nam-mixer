#!/bin/bash
cd "$(dirname "$0")"
D=$(cd ../work/cg/jcm800 && pwd)
for n in 10 3 5 2; do
  ../.venv-a2/bin/python single_nam_train.py $D/cg_$n --epochs 60 --name JCM800_ContinuousGain_${n}Captures > $D/train_$n.log 2>&1
done
