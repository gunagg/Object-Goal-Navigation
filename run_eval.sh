#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat

cd /srv/share3/rramrakhya6/Object-Goal-Navigation
echo "Starting eval"
echo "hab sim: ${PYTHONPATH}"

path=$1
python evaluate.py --agent o_nav --split val --eval 1 --num_eval_episodes 1000 --auto_gpu_config 0 --load $path