#!/bin/bash
source ~/.bashrc
conda activate venv
python testbaselines_tubetk.py --config configs/tubetk/mrabifurc.json --vesselfunc frangi --batch_size 40 --num_cores 16
python testbaselines_tubetk.py --config configs/tubetk/mrabifurc.json --vesselfunc meijering --batch_size 40 --num_cores 16
python testbaselines_tubetk.py --config configs/tubetk/mrabifurc.json --vesselfunc sato --batch_size 40 --num_cores 16
python testbaselines_tubetk.py --config configs/tubetk/mrabifurc.json --vesselfunc hessian --batch_size 40 --num_cores 16
