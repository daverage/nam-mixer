#!/bin/bash
# Phase 4E: train the 8 frozen models (A/B x 2 amps x seeds 0,1) concurrently with the stock official trainer wrapper. Records wall-clock per model.
cd "$(dirname "$0")"
B=$(cd .. && pwd)/work/p4e
mkdir -p $B/logs
for amp in jcm800 vibrolux; do for cfg in A B; do for seed in 0 1; do
  (
    s=$(date +%s)
    SINGLE_NAM_AMP=$amp ../.venv-a2/bin/python single_nam_train.py $B/$amp/${cfg}_bundle --epochs 60 --seed $seed --name ${amp}_P4E_${cfg}_s${seed} > $B/logs/train_${amp}_${cfg}_s${seed}.log 2>&1
    e=$(date +%s); echo "{\"amp\":\"$amp\",\"config\":\"$cfg\",\"seed\":$seed,\"wall_seconds\":$((e-s)),\"concurrent_models\":8}" > $B/logs/wall_${amp}_${cfg}_s${seed}.json
  ) &
done; done; done
wait
