#!/bin/bash
# Final-candidate task: train the 4 frozen models (FC x 2 amps x seeds 0,1) concurrently with the stock official trainer wrapper (scripts/single_nam_train.py). Records wall-clock.
cd "$(dirname "$0")"
B=$(cd .. && pwd)/work/p4e/final; mkdir -p $B/logs
for amp in jcm800 vibrolux; do for seed in 0 1; do
  ( s=$(date +%s)
    SINGLE_NAM_AMP=$amp ../.venv-a2/bin/python single_nam_train.py $B/$amp/FC_bundle --epochs 60 --seed $seed --name ${amp}_FC_s${seed} > $B/logs/train_${amp}_FC_s${seed}.log 2>&1
    e=$(date +%s); echo "{\"amp\":\"$amp\",\"seed\":$seed,\"wall_seconds\":$((e-s)),\"concurrent_models\":4}" > $B/logs/wall_${amp}_FC_s${seed}.json ) &
done; done
wait
