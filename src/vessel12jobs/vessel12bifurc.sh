#!/bin/bash
source ~/.bashrc
conda activate venv
python train.py -c configs/vessel12/vessel12bifurc.json --run_id v1
