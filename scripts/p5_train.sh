#!/bin/bash
# Expansion: train the six first-pass FC models (seed 0 only) concurrently with the unchanged official trainer wrapper. Records wall-clock.
cd "$(dirname "$0")"; B=$(cd .. && pwd)/work/p4e/final; mkdir -p $B/logs
for amp in twin supersonic peavey peavey6505 mesa orange; do
  ( s=$(date +%s)
    SINGLE_NAM_AMP=$amp ../.venv-a2/bin/python single_nam_train.py $B/$amp/FC_bundle --epochs 60 --seed 0 --name ${amp}_FC_s0 > $B/logs/train_${amp}_FC_s0.log 2>&1
    e=$(date +%s); echo "{\"amp\":\"$amp\",\"seed\":0,\"wall_seconds\":$((e-s)),\"concurrent_models\":6}" > $B/logs/wall_${amp}_FC_s0.json ) &
done; wait
