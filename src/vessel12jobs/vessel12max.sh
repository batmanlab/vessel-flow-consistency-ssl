#!/bin/bash
source ~/.bashrc
conda activate venv
python train.py -c configs/vessel12/vessel12max.json --run_id v2 --resume /ocean/projects/asc170022p/rohit33/selfsupmodels/saved/models/Vessel12Bifurc/v1/checkpoint-epoch5.pth
