# Flow based consistencies for self-supervised vessel segmentation

This is the code for the MICCAI 2021 paper **Self-Supervised Vessel Enhancement Using Flow-Based Consistencies**.

## Installation
Install the packages from `requirements.txt` file. In a virtual environment, use `pip` to install the required packages:
```
pip install -r requirements.txt
```

## Training
To run each of these scripts, go to the `src` directory.
### DRIVE dataset
To train a model to predict *vesselness* (without considering bifurcations) use the following script:
```
python train.py -c configs/v1/drivesq4_1110.json --run_id DRIVE
```
Other configurations like `configs/v1/drivesq4_1101.json`, `configs/v1/drivesq4_10.110.json`, etc. can also be tried. Refer to the configuration files in `configs/v1` for more details.

For inference of the trained model on DRIVE/STARE dataset, use the following script:
```
python test.py -c configs/v1/drivesq4_1110.json --run_id DRIVE_test --resume /path/to/ckpt.pth --train [1/0]
python test.py -c configs/v1/starefinetune.json --run_id STARE_test --resume /path/to/ckpt.pth --train [1/0]
```
The `--train` parameter specifies if you wish to save the vesselness files for training or test set. This is important because the training set is used to determine the threshold, and then evaluation is done with this threshold on the test set.

### Bifurcations
To train a model to predict *vesselness* (considering bifurcations) use the following script:
```
python train.py -c configs/bifurcv1/drivesq4bifurconly_1110.json --run_id DRIVE_bifurc
```
For inference of the trained model, use the same configuration file (since the configuration files specify the loss function and vesselness functions to use).
```
python test.py -c configs/bifurcv1/drivesq4bifurconly_1110.json --run_id DRIVE_bifurc_test --resume /path/to/ckpt.pth --train [1/0]
```

When `test.py` runs, it asks for a path to save the vesselness and other outputs. Specify a filename, and save it. Now, use this saved model to get evaluation metrics. Use the following script:
```
python get_all_metrics_drivestare.py -c configs/v1/drivesq4_1110.json --run_id DRIVE_test --resume /path/to/ckpt.pth
```



### Important files:
* `get_all_metrics_drivestare.py` : Contains code for obtaining evaluation metrics on the DRIVE and STARE datasets. Compares with other 2D vesselness methods. Contains code for obtaining similar metrics for bifurcation locations (locations are annotated manually).
* `get_all_metrics_hrf.py` : Similar as before, but for the HRF dataset.


## TODO
- [ ] File names need to be changed at a lot of places (preferably use a master config file for storing common items like data and save directories, etc.).