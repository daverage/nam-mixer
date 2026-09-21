#!/bin/bash
# Expansion: build the six FC bundles (unchanged scripts/fc_build.py) from docs/expand/manifest_frozen.json, in parallel.
cd "$(dirname "$0")"; B=$(cd .. && pwd)/work/p4e/final; mkdir -p $B/logs
for amp in twin supersonic peavey peavey6505 mesa orange; do
  ( args=$(../.venv-a2/bin/python -c "
import json;m=json.load(open('../docs/expand/manifest_frozen.json'))['amps']['$amp']
print('--gains',*m['selected_positions'],'--anchors',*m['anchors_input_gain_db'])")
    mkdir -p $B/$amp; s=$(date +%s)
    ../.venv-a2/bin/python fc_build.py $amp $args --offsets -32 -24 -16 -8 0 8 14 20 --val-offsets -24 0 8 20 --di-seconds 20 --val-seconds 8 --out $B/$amp/FC_bundle > $B/logs/build_$amp.log 2>&1
    echo "{\"amp\":\"$amp\",\"build_seconds\":$(( $(date +%s)-s ))}" > $B/logs/build_wall_$amp.json ) &
done; wait
