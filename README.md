# Flow based consistencies for self-supervised vessel segmentation

This is the code for the MICCAI 2021 paper **Self-Supervised Vessel Enhancement Using Flow-Based Consistencies**.

## Installation
Install the packages from `requirements.txt` file. In a virtual environment, use `pip` to install the required packages:
```
pip install -r requirements.txt
```

---

## Training
To run each of these scripts, go to the `src` directory.

### DRIVE/STARE dataset
To train a model to predict *vesselness* (without considering bifurcations) use the following script:
```
python train.py -c configs/v1/drivesq4_1110.json --run_id DRIVE
```
Other configurations like `configs/v1/drivesq4_1101.json`, `configs/v1/drivesq4_10.110.json`, etc. can also be used. Refer to the configuration files in `configs/v1` for more configurations (or create your own!).


For inference of the trained model on DRIVE/STARE dataset, use the following script:
```
python test.py -c configs/v1/drivesq4_1110.json --run_id DRIVE_test --resume /path/to/ckpt.pth --train [1/0]
python test.py -c configs/v1/starefinetune.json --run_id STARE_test --resume /path/to/ckpt.pth --train [1/0]
```

The `--train` parameter specifies if you wish to save the vesselness files for training or test set. This is important because the training set is used to determine the threshold, and then evaluation is done with this threshold on the test set.

---

### Bifurcations
To train a model to predict *vesselness* (considering bifurcations) use the following script:
```
python train.py -c configs/bifurcv1/drivesq4bifurconly_1110.json --run_id DRIVE_bifurc
```
For inference of the trained model, use the same configuration file (since the configuration files specify the loss function and vesselness functions to use).
```
python test.py -c configs/bifurcv1/drivesq4bifurconly_1110.json --run_id DRIVE_bifurc_test --resume /path/to/ckpt.pth --train [1/0]
```

---

### RITE/HRF dataset
Similar to DRIVE dataset, you can use the following script for training: 
```
python train.py -c configs/rite/ritebifurc.json --run_id RITE
python train.py -c configs/hrf/hrfbifurc.json --run_id HRF
```

To test the method, use the following scripts
```
python test.py -c configs/rite/ritebifurc.json --run_id RITE_test --path /path/to/ckpt.pth --train [1/0]
python test.py -c configs/hrf/hrfbifurc.json --run_id HRF_test --path /path/to/ckpt.pth --train [1/0]
```

---

### 3D datasets
To train a model to predict *vesselness* in 3D datasets like COPDGene, VESSEL12, VascuSynth and TubeTK datasets, use the following scripts:
```
python train.py -c configs/copd/copdsq4.json --run_id COPDGene
python train.py -c configs/vessel12/vessel12bifurc.json --run_id VESSEL12 
python train.py -c configs/tubetk/mrabifurc.json --run_id TubeTK
python train.py -c configs/vascu/vascusigma0.json --run_id VascuSynth
```

Testing is also similar to the 2D datasets:
```
python testcopd3d.py -c configs/copd/copdsq4.json --run_id COPDGene_test --resume /path/to/ckpt.pth --patientIDs <list of patientIDs>
python test3d.py -c configs/vessel12/vessel12bifurc.json --run_id VESSEL12_test --resume /path/to/ckpt.pth
python test3d.py -c configs/tubetk/mrabifurc.json --run_id TubeTK_test --resume /path/to/ckpt.pth
python test3d.py -c configs/vascu/vascusigma0.json --run_id VascuSynth_test --resume /path/to/ckpt.pth
```
The COPD test also takes a list of patient IDs because there are over 9000 patient volumes.


For 3D datasets, we also save results from other baselines (Frangi, Sato, etc.):
```
python testbaselines_vessel12.py --config configs/vessel12/vessel12bifurc.json --vesselfunc <method> --batch_size 40 --num_cores 16
python testbaselines_tubetk.py --config configs/tubetk/mrabifurc.json --vesselfunc <method> --batch_size 40 --num_cores 16
```


---

## Evaluate metrics

### 2D datasets (DRIVE/STARE/RITE/HRF)
When `test.py` runs, it asks for a path to save the vesselness and other outputs. Specify a filename, and save it. Now, use this saved model to get evaluation metrics. Use the following script:
```
python get_all_metrics_drivestare.py -c configs/v1/drivesq4_1110.json --run_id DRIVE_test --resume /path/to/ckpt.pth --method <method>
python get_all_metrics_[rite|hrf].py -c /path/to/training_config.json --run_id [RITE|HRF]_test --resume /path/to/ckpt.pth --method <method>
```
Use the `--method` parameter to specify which vesselness method to use for evaluation (choices: `frangi`, `sato`, `hessian`, `meijering`, `ours`).

---

## Reference
If you use our work in your research or wish to refer to the results in our paper, please use the following BibTex reference:

```BibTeX
@article{jena2021self,
  title={Self-supervised vessel enhancement using flow-based consistencies},
  author={Jena, Rohit and Singla, Sumedha and Batmanghelich, Kayhan},
  journal={arXiv preprint arXiv:2101.05145},
  year={2021}
}
```

## TODO
- [ ] Add paths to pretrained models.
- [ ] File names need to be changed at a lot of places (preferably use a master config file for storing common items like data and save directories, etc.).