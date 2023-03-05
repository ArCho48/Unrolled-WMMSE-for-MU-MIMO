
# UWMMSE-MIMO
Tensorflow implementation of UWMMSE-MIMO

## Overview
This library contains a Tensorflow implementation of multi-antenna version of UWMMSE[1].
## Dependencies

* **python>=3.6**
* **tensorflow>=1.14.0**: https://tensorflow.org
* **numpy**
* **matplotlib**

## Structure
* [main](https://github.com/ArCho48/UWMMSE-MIMO/blob/master/main.py): Main code for running the experiments in the paper. Run as python3 main.py --datasetID {dataset ID} --tx_antennas {T} --rx_antennas {R} --expID {exp ID} --mode {mode}. For ex. to train UWMMSE on dataset with ID set3 having 5 tx and 3 rx antennas, run python3 main.py --datasetID set3 --tx_antennas 5 --rx_antennas 3 --expID uwmmse --mode train.
* [model](https://github.com/ArCho48/UWMMSE-MIMO/blob/master/model.py): Defines the UWMMSE model.
* [data](https://github.com/ArCho48/UWMMSE-MIMO/tree/master/data): should contain your dataset in folder {dataset ID}. 
* [models](https://github.com/ArCho48/UWMMSE-MIMO/tree/master/models): Stores trained models in a folder with same name as {datset ID}.
* [results](https://github.com/ArCho48/UWMMSE-MIMO/tree/master/results): Stores results in a folder with same name as {datset ID}.

## Usage


Please cite [[1](#citation)] in your work when using this library in your experiments.

## Feedback
For questions and comments, feel free to contact [Arindam Chowdhury](mailto:arindam.chowdhury@rice.edu).

## Citation
```
[1] Chowdhury A, Verma G, Rao C, Swami A, Segarra S. ML-aided power allocation for Tactical MIMO. 
arXiv preprint arXiv:2109.06992 2021 Sep 14.
```

BibTeX format:
```
@article{chowdhury2021ml,
  title={ML-aided power allocation for Tactical MIMO},
  author={Chowdhury, Arindam and Verma, Gunjan and Rao, Chirag and Swami, Ananthram and Segarra, Santiago},
  journal={arXiv e-prints},
  pages={arXiv--2109},
  year={2021}
}
```
