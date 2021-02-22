#!/bin/bash
method=$1
config=$2
ckpt=$3

folder=${method}-$config-$ckpt

echo python test.py -c /pghbio/dbmi/batmanlab/rohit33/saved_models/models/$method/v1-2d-$config/config.json --resume /pghbio/dbmi/batmanlab/rohit33/saved_models/models/$method/v1-2d-$config/checkpoint-epoch${ckpt}.pth --crosscorr 0 --train 1 > /dev/null
python test.py -c /pghbio/dbmi/batmanlab/rohit33/saved_models/models/$method/v1-2d-$config/config.json --resume /pghbio/dbmi/batmanlab/rohit33/saved_models/models/$method/v1-2d-$config/checkpoint-epoch${ckpt}.pth --crosscorr 0 --train 1 > /dev/null
echo python test.py -c /pghbio/dbmi/batmanlab/rohit33/saved_models/models/$method/v1-2d-$config/config.json --resume /pghbio/dbmi/batmanlab/rohit33/saved_models/models/$method/v1-2d-$config/checkpoint-epoch${ckpt}.pth --crosscorr 0 --train 0 > /dev/null
python test.py -c /pghbio/dbmi/batmanlab/rohit33/saved_models/models/$method/v1-2d-$config/config.json --resume /pghbio/dbmi/batmanlab/rohit33/saved_models/models/$method/v1-2d-$config/checkpoint-epoch${ckpt}.pth --crosscorr 0 --train 0 > /dev/null

mkdir -p ${folder}
mv test_analysis.pkl ${folder}/
mv train_analysis.pkl ${folder}/
mv test_vesselness.pkl ${folder}/
mv train_vesselness.pkl ${folder}/
