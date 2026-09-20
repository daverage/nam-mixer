export SINGLE_NAM_AMP=mesa
P=../.venv-a2/bin/python
$P single_nam_build_dataset.py --gains 1 5 10 --name method_a_3 && $P single_nam_train.py method_a_3 --epochs 60 --name Mesa_ContinuousGain_3Captures
$P single_nam_build_dataset.py --name method_a_10 && $P single_nam_train.py method_a_10 --epochs 60 --name Mesa_ContinuousGain_10Captures
echo ALLDONE
