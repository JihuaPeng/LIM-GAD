# LIM-GAD
Language-Instructed Multimodal Learning for Interpretable Group Activity Detection


## Dependencies
Make sure you have the following dependencies installed:
* Ubuntu 20.04
* CUDA 11.2
* Python 3.8
* PyTorch 1.8.1
* Matplotlib=3.1.0

## Datasets
Our model is evaluated on [CAFE](https://dk-kim.github.io/CAFE/).


## Training from scratch
python train.py --data_path Dataset/ --split 'view' --batch 4 --test_batch 4 --group_ce_loss_coef 2 --group_code_loss_coef 2
