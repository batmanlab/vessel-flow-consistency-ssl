#!/bin/bash
source ~/.bashrc
conda activate venv
echo "Hello"
python test_vessel12.py --config configs/vessel12/vessel12bifurc.json --resume /ocean/projects/asc170022p/rohit33/selfsupmodels/saved/models/Vessel12Bifurc/v2/checkpoint-epoch2.pth --run_id test --batch_size 20
python test_tubetk.py --config configs/tubetk/mrabifurc.json --resume /ocean/projects/asc170022p/rohit33/selfsupmodels/saved/models/TubeTK_MRA_Bifurc/v2/checkpoint-epoch2.pth --run_id test --batch_size 20
