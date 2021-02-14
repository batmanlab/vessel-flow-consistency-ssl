#!/bin/bash
source ~/.bashrc
conda activate venv
python train.py -c configs/rite/ritebifurc.json --run_id v1
