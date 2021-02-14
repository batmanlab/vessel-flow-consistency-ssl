#!/bin/bash
source ~/.bashrc
conda activate venv
python train.py -c configs/hrf/hrfbifurc.json --run_id v1
