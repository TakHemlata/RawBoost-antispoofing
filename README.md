RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing
===============
This repository contains our implementation of the paper, "RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing". This work introduce RawBoost, a data boosting and augmentation method for the design of more reliable spoofing detection solutions which operate directly upon raw waveform inputs ([Paper link here](https://arxiv.org/pdf/2111.04433.pdf)).


## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ git clone https://github.com/TakHemlata/RawBoost-antispoofing.git
$ conda create --name RawBoost_antispoofing python=3.8.8
$ conda activate RawBoost_antispoofing
$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
$ pip install -r requirements.txt
```


## Experiments

### Dataset
Our experiments are performed on the logical access (LA) partition of the ASVspoof 2021 dataset (train on 2019 LA training and evaluate on 2021 LA evaluation database).

### Training
To train the model run:
```
python main.py --track=LA --loss=WCE   --lr=0.0001 --batch_size=128
```

### Testing

To evaluate your own model on LA evaluation dataset:

```
python main.py --track=LA --loss=WCE --is_eval --eval --model_path='/path/to/your/best_model.pth' --eval_output='eval_CM_scores_file.txt'
```

We also provide a pre-trained models. To use it you can run: 
```
python main.py --track=LA --loss=WCE --is_eval --eval --model_path='Pre_trained_models.pth' --eval_output='RawBoost_eval_CM_scores.txt'
```

This repository is built on our End-to-end RawNet2 CM system (ASVspoof2021 Challenge baseline).
- [ASVspoof 2021 Challenge baseline repo](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2)


## Contact
For any query regarding this repository, please contact:
- Hemlata Tak: tak[at]eurecom[dot]fr
- Massimiliano Todisco: todisco[at]eurecom[dot]fr

## Citation
If you use RawBoost code in your research please use the following citation:

```bibtex
@inproceedings{tak2021rawboost,
  title={RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing},
  author={Tak, Hemlata and Kamble, Madhu and Patino, Jose and Todisco, Massimiliano and Evans, Nicholas},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2022}
}
```

