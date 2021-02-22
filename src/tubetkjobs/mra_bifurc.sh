#!/bin/bash
source ~/.bashrc
conda activate venv
python train.py -c configs/tubetk/mrabifurc.json --run_id v2 --resume /ocean/projects/asc170022p/rohit33/selfsupmodels/saved/models/TubeTK_MRA_Bifurc/v1/checkpoint-epoch10.pth
