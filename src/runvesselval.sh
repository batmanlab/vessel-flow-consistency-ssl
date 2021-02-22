#!/bin/bash
source ~/.bashrc
conda activate venv
python testbaselines_vessel12.py --config configs/vessel12/vessel12bifurc.json --vesselfunc frangi --batch_size 40 --num_cores 16
python testbaselines_vessel12.py --config configs/vessel12/vessel12bifurc.json --vesselfunc sato --batch_size 40 --num_cores 16
python testbaselines_vessel12.py --config configs/vessel12/vessel12bifurc.json --vesselfunc meijering --batch_size 40 --num_cores 16
python testbaselines_vessel12.py --config configs/vessel12/vessel12bifurc.json --vesselfunc hessian --batch_size 40 --num_cores 16
