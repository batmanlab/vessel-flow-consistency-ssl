#!/bin/bash
source ~/.bashrc
conda activate venv
python train.py -c configs/tubetk/mrabifurc.json --run_id v1
