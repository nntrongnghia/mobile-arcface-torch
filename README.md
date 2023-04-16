# Face Verification for Mobile
Train face verification model with ArcFace implemented with PyTorch. Models are intended to be used on mobile devices.

## Install

This repo is tested with Python 3.7 and PyTorch 1.13. Other packages' versions can be found in `requirements.txt`.

## Setup the dataset
This project use 2 datasets:
- [MS1Mv3 (also called MS1M-RetinaFace)](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface) for training
- [LFW (deep funneled)](http://vis-www.cs.umass.edu/lfw/) for validation

### MS1Mv3 for training
- Download and extract the dataset [here](https://drive.google.com/file/d/1JgmzL9OLTqDAZE86pBgETtSQL4USKTFy/view)
- In the config file `configs/ms1mv3_mbf.py`, modify the `config.root_dir` value with the path to the extracted folder containing the MS1Mv3 dataset.

### LFW for validation
- Download and extract the deep funneled version [here](http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz). In the config file `configs/ms1mv3_mbf.py`, modify the `img_dir` value of `config.lfwpair_kwargs` with the path to the extracted folder containing LFW images.

- Download the `pairs.txt` [here](http://vis-www.cs.umass.edu/lfw/pairs.txt). In the config file `configs/ms1mv3_mbf.py`, modify the `pairs_txt_path` value of `config.lfwpair_kwargs` with the path to the `pairs.txt` file.

## Train
Run `python train.py`. You can modify the `config.batch_size` in the config file to fit your GPU memory if needed.

Then run `tensorboard --logdir ./lightning_logs` to view the training progress in realtime.

The best checkpoint will be saved in the corressponding folder in `./lightning_logs/`.
### Metrics logged during training
- `mean_target_cosine`: Training metric - Mean value in each batch of cosine similarity between the image and the target class cluster center
- `train_loss`: ArcFace loss
- `lfw_auroc`: Validation metric - AUC ROC based on predictions of LFW pair images.
- `pos_mean_score`: Validation metric - Mean similarity score of all matching pair images
- `neg_mean_score`: Validation metric - Mean similarity score of all mismatch pair images