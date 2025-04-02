# Project README

## Announcement:
This project is the official implementation of the code for the paper titled "METC: A Hybrid Deep Learning Framework for Cross-Network Encrypted DNS over
HTTPS Traffic Detection and Tunnel Identification". If you use this code or find it helpful in your research, please cite the paper:
M. Zuo, C. Guo, H. Xu et al., METC: A Hybrid Deep Learning Framework for Cross-Network Encrypted DNS over HTTPS Traffic Detection and Tunnel Identification, Information Fusion (2025), doi: https://doi.org/10.1016/j.inffus.2025.103125.
```
@article{ZUO2025103125,
title = {METC: A Hybrid Deep Learning Framework for Cross-Network Encrypted DNS over HTTPS Traffic Detection and Tunnel Identification},
journal = {Information Fusion},
volume = {121},
pages = {103125},
year = {2025},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2025.103125},
url = {https://www.sciencedirect.com/science/article/pii/S1566253525001988},
author = {Ming Zuo and Changyong Guo and Haiyan Xu and Zhaoxin Zhang and Yanan Cheng},
keywords = {DNS over HTTPS (DoH), Encrypted Traffic Detection, Tunnel Identification, Deep Learning, Machine Learning, Network Security},
}
```
....


## Overview

This project provides a unified framework for **training**, **fine-tuning**, and **testing** machine learning models using various architectures like CNNs, LSTMs, Random Forests, and XGBoost. The setup supports multiple CUDA versions, enabling performance optimization on NVIDIA GPUs.

---

## Installation

### Environment Setup
Choose the appropriate installation command based on your CUDA version. For the latest features, install **PyTorch 2.4.1** and its dependencies.

#### Using Conda
```shell
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 -c pytorch
```
#### Using Pip
```shell
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
```
---

### CUDA Compatibility

#### CUDA 11.8
```shell
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```
```shell
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```
#### CUDA 12.1
```shell
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
```shell
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```
#### CUDA 12.4
```shell
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```
```shell
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```
---

## Usage
### Dataset
- CIRA-CIC-DoHBrw-2020
- doh_dataset  
- Custom_dataset
- FiveWeek
- Generated
- RealWorld
- Tunnel

python main.py --model_type CNN_BiGRU_Attention --test --test_data_dir ../csv_output/doh_dataset --test_checkpoint_path CNN_BiGRU_Attention_best_model_checkpoint.pth

python main.py --model_type XGBoost --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path XGBoost_best_model_checkpoint.pkl

XGBoost: 
[CV 3/3] END colsample_bytree=0.8, gamma=0.1, learning_rate=0.001, max_depth=10, n_estimators=100, subsample=0.8;, score=0.995 total time= 3.1min
[CV 1/3] END colsample_bytree=0.8, gamma=0.1, learning_rate=0.001, max_depth=10, n_estimators=100, subsample=0.8;, score=0.995 total time= 4.2min
[CV 1/3] END colsample_bytree=0.8, gamma=0.1, learning_rate=0.001, max_depth=10, n_estimators=100, subsample=0.8;, score=0.995 total time= 4.2min
[CV 1/3] END colsample_bytree=0.8, gamma=0.1, learning_rate=0.001, max_depth=10, n_estimators=100, subsample=1.0;, score=0.995 total time= 3.7min
[CV 3/3] END colsample_bytree=0.8, gamma=0.1, learning_rate=0.001, max_depth=10, n_estimators=100, subsample=1.0;, score=0.995 total time= 3.4min
[CV 2/3] END colsample_bytree=0.8, gamma=0.1, learning_rate=0.001, max_depth=10, n_estimators=100, subsample=1.0;, score=0.995 total time= 4.6min
[CV 1/3] END colsample_bytree=0.8, gamma=0.1, learning_rate=0.001, max_depth=10, n_estimators=1100, subsample=1.0;, score=0.995 total time= 5.3min
[CV 3/3] END colsample_bytree=0.8, gamma=0.1, learning_rate=0.001, max_depth=10, n_estimators=1100, subsample=0.8;, score=0.995 total time= 5.4min
[CV 1/3] END colsample_bytree=0.8, gamma=0.1, learning_rate=0.001, max_depth=10, n_estimators=1100, subsample=0.8;, score=0.995 total time= 6.0min
[CV 3/3] END colsample_bytree=0.8, gamma=0.1, learning_rate=0.001, max_depth=10, n_estimators=200, subsample=0.8;, score=0.995 total time= 7.3min
[CV 1/3] END colsample_bytree=0.8, gamma=0.1, learning_rate=0.01, max_depth=5, n_estimators=1100, subsample=0.8;, score=0.986 total time= 2.9min
[CV 2/3] END colsample_bytree=0.8, gamma=0.1, learning_rate=0.001, max_depth=10, n_estimators=200, subsample=1.0;, score=0.995 total time= 6.9min
[CV 1/3] END colsample_bytree=0.8, gamma=0.1, learning_rate=0.001, max_depth=10, n_estimators=200, subsample=1.0;, score=0.995 total time= 7.6min

RandomForest   XGBoost
NOT RUNNING:
'CNN_Attention', 'BiGRU_Attention', 'BiLSTM_Attention'


```shell
jobs -p | xargs kill
```
nohup python main.py --model_type LogisticRegression --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type AdaBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type DecisionTree --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type NaiveBayes --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type LDA --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type ExtraTrees --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type CatBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type LightGBM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type RandomForest --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type XGBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type CNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type RNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type DNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type MLP --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type BiGRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type BiLSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type CNN_GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type CNN_LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type CNN_GRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type CNN_LSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type CNN_BiGRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &
nohup python main.py --model_type CNN_BiLSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 &


nohup python main.py --model_type LogisticRegression --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type AdaBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type DecisionTree --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type NaiveBayes --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type LDA --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type ExtraTrees --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type CatBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type LightGBM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type RandomForest --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type XGBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type CNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type RNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type DNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type MLP --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type BiGRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type BiLSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type CNN_GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type CNN_LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type CNN_GRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type CNN_LSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type CNN_BiGRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &
nohup python main.py --model_type CNN_BiLSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/doh_dataset &



nohup python main.py --model_type LogisticRegression --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type AdaBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type DecisionTree --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type NaiveBayes --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type LDA --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type ExtraTrees --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type CatBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type LightGBM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type RandomForest --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type XGBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type CNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type RNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type DNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type MLP --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type BiGRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type BiLSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type CNN_GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type CNN_LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type CNN_GRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type CNN_LSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type CNN_BiGRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &
nohup python main.py --model_type CNN_BiLSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Custom_dataset &



nohup python main.py --model_type LogisticRegression --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type AdaBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type DecisionTree --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type NaiveBayes --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type LDA --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type ExtraTrees --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type CatBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type LightGBM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type RandomForest --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type XGBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type CNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type RNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type DNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type MLP --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type BiGRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type BiLSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type CNN_GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type CNN_LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type CNN_GRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type CNN_LSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type CNN_BiGRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &
nohup python main.py --model_type CNN_BiLSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Generated &


nohup python main.py --model_type LogisticRegression --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type AdaBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type DecisionTree --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type NaiveBayes --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type LDA --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type ExtraTrees --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type CatBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type LightGBM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type RandomForest --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type XGBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type CNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type RNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type DNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type MLP --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type BiGRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type BiLSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type CNN_GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type CNN_LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type CNN_GRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type CNN_LSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type CNN_BiGRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &
nohup python main.py --model_type CNN_BiLSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/FiveWeek &



nohup python main.py --model_type LogisticRegression --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type AdaBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type DecisionTree --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type NaiveBayes --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type LDA --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type ExtraTrees --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type CatBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type LightGBM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type RandomForest --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type XGBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type CNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type RNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type DNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type MLP --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type BiGRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type BiLSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type CNN_GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type CNN_LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type CNN_GRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type CNN_LSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type CNN_BiGRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &
nohup python main.py --model_type CNN_BiLSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/RealWorld &

####################################################################################################################################################


nohup python main.py --model_type LogisticRegression --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type AdaBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type DecisionTree --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type NaiveBayes --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type LDA --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type ExtraTrees --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type CatBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type LightGBM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type RandomForest --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type XGBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type CNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type RNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type DNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type MLP --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type BiGRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type BiLSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type CNN_GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type CNN_LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type CNN_GRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type CNN_LSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type CNN_BiGRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &
nohup python main.py --model_type CNN_BiLSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel &


nohup python main.py --model_type LogisticRegression --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type AdaBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type DecisionTree --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type NaiveBayes --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type LDA --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type ExtraTrees --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type CatBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type LightGBM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type RandomForest --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type XGBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type CNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type RNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type DNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type MLP --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type BiGRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type BiLSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type CNN_GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type CNN_LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type CNN_GRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type CNN_LSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type CNN_BiGRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &
nohup python main.py --model_type CNN_BiLSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel &



nohup python main.py --model_type LogisticRegression --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type AdaBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type DecisionTree --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type NaiveBayes --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type LDA --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type ExtraTrees --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type CatBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type LightGBM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type RandomForest --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type XGBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type CNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type RNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type DNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type MLP --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type BiGRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type BiLSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type CNN_GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type CNN_LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type CNN_GRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type CNN_LSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type CNN_BiGRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &
nohup python main.py --model_type CNN_BiLSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD &


nohup python main.py --model_type LogisticRegression --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type AdaBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type DecisionTree --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type NaiveBayes --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type LDA --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type ExtraTrees --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type CatBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type LightGBM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type RandomForest --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type XGBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type CNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type RNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type DNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type MLP --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type BiGRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type BiLSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type CNN_GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type CNN_LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type CNN_GRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type CNN_LSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type CNN_BiGRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &
nohup python main.py --model_type CNN_BiLSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel/DoH-Tunnel-Traffic-HKD &


### Training
Train a model using the specified architecture and dataset.

#### Example Commands
```shell
python main.py --model_type XGBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/Tunnel
```
```shell
python main.py --model_type XGBoost --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020
```
```shell
python main.py --model_type RandomForest --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020
```
```shell
python main.py --model_type CNN --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020
```

```shell
python main.py --model_type LSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020
```

```shell
python main.py --model_type GRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020
```

```shell
python main.py --model_type BiLSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020
```

```shell
python main.py --model_type BiGRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020
```
```shell
python main.py --model_type CNN_BiLSTM_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020
```
```shell
python main.py --model_type CNN_BiGRU_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020
```
```shell
python main.py --model_type CNN_BiLSTM --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020
```
```shell
python main.py --model_type CNN_BiGRU --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020
```
```shell
python main.py --model_type CNN_Attention --train --epochs 100 --batch_size 256 --data_dir ../csv_output/CIRA-CIC-DoHBrw-2020
```
---

### Fine-Tuning
Fine-tune a pre-trained model on a new dataset.


python main.py --model_type CNN_BiGRU_Attention --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 20 --best_checkpoint_path CNN_BiGRU_Attention_best_model_checkpoint.pth --sample_size 0.3
python main.py --model_type CNN_BiGRU_Attention --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path CNN_BiGRU_Attention_fine_tuned_best_model.pth

python main.py --model_type XGBoost --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 20 --best_checkpoint_path XGBoost_best_model_checkpoint.pkl --sample_size 0.3
python main.py --model_type XGBoost --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path XGBoost_fine_tuned_model.pkl

python main.py --model_type CNN --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path CNN_best_model_checkpoint.pth --sample_size 0.1
python main.py --model_type CNN --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path CNN_fine_tuned_best_model.pth

python main.py --model_type LSTM --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path LSTM_best_model_checkpoint.pth --sample_size 0.1
python main.py --model_type LSTM --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path LSTM_fine_tuned_best_model.pth

python main.py --model_type GRU --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path GRU_best_model_checkpoint.pth --sample_size 0.1
python main.py --model_type GRU --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path GRU_fine_tuned_best_model.pth

python main.py --model_type BiLSTM --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path BiLSTM_best_model_checkpoint.pth --sample_size 0.1
python main.py --model_type BiLSTM --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path BiLSTM_fine_tuned_best_model.pth

python main.py --model_type BiGRU --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path BiGRU_best_model_checkpoint.pth --sample_size 0.1
python main.py --model_type BiGRU --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path BiGRU_fine_tuned_best_model.pth

python main.py --model_type RNN --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path RNN_best_model_checkpoint.pth --sample_size 0.1
python main.py --model_type RNN --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path RNN_fine_tuned_best_model.pth

python main.py --model_type DNN --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path DNN_best_model_checkpoint.pth --sample_size 0.1
python main.py --model_type DNN --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path DNN_fine_tuned_best_model.pth

python main.py --model_type MLP --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path MLP_best_model_checkpoint.pth --sample_size 0.1
python main.py --model_type MLP --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path MLP_fine_tuned_best_model.pth

python main.py --model_type CNN_GRU --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path CNN_GRU_best_model_checkpoint.pth --sample_size 0.1
python main.py --model_type CNN_GRU --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path CNN_GRU_fine_tuned_best_model.pth

python main.py --model_type CNN_LSTM --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path CNN_LSTM_best_model_checkpoint.pth --sample_size 0.1
python main.py --model_type CNN_LSTM --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path CNN_LSTM_fine_tuned_best_model.pth

python main.py --model_type CNN_GRU_Attention --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path CNN_GRU_Attention_best_model_checkpoint.pth --sample_size 0.1
python main.py --model_type CNN_GRU_Attention --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path CNN_GRU_Attention_fine_tuned_best_model.pth

python main.py --model_type CNN_LSTM_Attention --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path CNN_LSTM_Attention_best_model_checkpoint.pth --sample_size 0.1
python main.py --model_type CNN_LSTM_Attention --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path CNN_LSTM_Attention_fine_tuned_best_model.pth

python main.py --model_type CNN_BiLSTM_Attention --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path CNN_BiLSTM_Attention_best_model_checkpoint.pth --sample_size 0.1
python main.py --model_type CNN_BiLSTM_Attention --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path CNN_BiLSTM_Attention_fine_tuned_best_model.pth

python main.py --model_type CNN_BiLSTM --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path CNN_BiLSTM_best_model_checkpoint.pth --sample_size 0.1
python main.py --model_type CNN_BiLSTM --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path CNN_BiLSTM_fine_tuned_best_model.pth

python main.py --model_type CNN_Attention --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path CNN_Attention_best_model_checkpoint.pth --sample_size 0.1
python main.py --model_type CNN_Attention --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path CNN_Attention_fine_tuned_best_model.pth

python main.py --model_type RandomForest --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path RandomForest_best_model_checkpoint.pkl --sample_size 0.1
python main.py --model_type RandomForest --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path RandomForest_fine_tuned_model.pkl

python main.py --model_type LogisticRegression --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path LogisticRegression_best_model_checkpoint.pkl --sample_size 0.1
python main.py --model_type LogisticRegression --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path LogisticRegression_fine_tuned_model.pkl

python main.py --model_type AdaBoost --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path AdaBoost_best_model_checkpoint.pkl --sample_size 0.1
python main.py --model_type AdaBoost --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path AdaBoost_fine_tuned_model.pkl

python main.py --model_type DecisionTree --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path DecisionTree_best_model_checkpoint.pkl --sample_size 0.1
python main.py --model_type DecisionTree --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path DecisionTree_fine_tuned_model.pkl

python main.py --model_type NaiveBayes --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path NaiveBayes_best_model_checkpoint.pkl --sample_size 0.1
python main.py --model_type NaiveBayes --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path NaiveBayes_fine_tuned_model.pkl

python main.py --model_type LDA --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path LDA_best_model_checkpoint.pkl --sample_size 0.1
python main.py --model_type LDA --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path LDA_fine_tuned_model.pkl

python main.py --model_type ExtraTrees --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path ExtraTrees_best_model_checkpoint.pkl --sample_size 0.1
python main.py --model_type ExtraTrees --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path ExtraTrees_fine_tuned_model.pkl

python main.py --model_type CatBoost --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path CatBoost_best_model_checkpoint.pkl --sample_size 0.1
python main.py --model_type CatBoost --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path CatBoost_fine_tuned_model.pkl

python main.py --model_type LightGBM --fine_tune --fine_tune_data_dir ../csv_output/RealWorld --fine_tune_epochs 10 --best_checkpoint_path LightGBM_best_model_checkpoint.pkl --sample_size 0.1
python main.py --model_type LightGBM --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path LightGBM_fine_tuned_model.pkl



#### Example Commands
```shell
python main.py --model_type CNN_BiLSTM_Attention --fine_tune --fine_tune_data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 --fine_tune_epochs 10 --best_checkpoint_path CNN_BiLSTM_Attention_best_model_checkpoint.pth
```
```shell
python main.py --model_type BiLSTM --fine_tune --fine_tune_data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 --fine_tune_epochs 10 --best_checkpoint_path BiLSTM_best_model_checkpoint.pth
````
```shell
python main.py --model_type RandomForest --fine_tune --fine_tune_data_dir ../csv_output/doh_dataset --fine_tune_epochs 10 --best_checkpoint_path RandomForest_model.pkl --sample_size 0.1
```
```shell
python main.py --model_type XGBoost --fine_tune --fine_tune_data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 --fine_tune_epochs 10 --best_checkpoint_path XGBoost_model.pkl
```
---
CNN_BiLSTM_best_model_checkpoint.pth
### Testing
Evaluate the performance of a trained or fine-tuned model on a test dataset(never used before).

#### Example Commands
```shell
python main.py --model_type CNN_BiLSTM --test --test_data_dir ../csv_output/Custom_dataset --test_checkpoint_path CNN_BiLSTM_best_model_checkpoint.pth
```
```shell
python main.py --model_type CNN_BiLSTM_Attention --test --test_data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 --test_checkpoint_path CNN_BiLSTM_Attention_fine_tuned_best_model.pth
```
```shell
python main.py --model_type BiLSTM --test --test_data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 --test_checkpoint_path BiLSTM_fine_tuned_model.pth
```
```shell
python main.py --model_type XGBoost --test --test_data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 --test_checkpoint_path XGBoost_fine_tuned_model.pkl
```
```shell
python main.py --model_type RandomForest --test --test_data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 --test_checkpoint_path RandomForest_fine_tuned_model.pkl
```
```shell
python main.py --model_type XGBoost --test --test_data_dir ../csv_output/RealWorld --test_checkpoint_path XGBoost_best_model_checkpoint.pkl
```
---


### Explain
Explain

#### Example Commands
```shell
python main.py --explain --model_type CNN_BiLSTM_Attention --explain_checkpoint_path CNN_BiLSTM_Attention_best_model_checkpoint.pth --explain_data_dir ../csv_output/doh_dataset --data_dir ../csv_output/doh_dataset
```
```shell
python main.py --explain --model_type CNN_BiGRU_Attention --explain_checkpoint_path CNN_BiLSTM_best_model_checkpoint.pth --explain_data_dir ../csv_output/doh_dataset --data_dir ../csv_output/doh_dataset
```
```shell
python main.py --explain --model_type CNN_BiLSTM --explain_checkpoint_path CNN_BiLSTM_best_model_checkpoint.pth --explain_data_dir ../csv_output/doh_dataset --data_dir ../csv_output/doh_dataset
```
```shell
python main.py --explain --model_type CNN_BiGRU --explain_checkpoint_path CNN_BiGRU_best_model_checkpoint.pth --explain_data_dir ../csv_output/doh_dataset --data_dir ../csv_output/doh_dataset
```
```shell
python main.py --explain --model_type BiLSTM --explain_checkpoint_path BiLSTM_best_model_checkpoint.pth --explain_data_dir ../csv_output/doh_dataset --data_dir ../csv_output/doh_dataset
```
```shell
python main.py --explain --model_type BiGRU --explain_checkpoint_path BiGRU_best_model_checkpoint.pth --explain_data_dir ../csv_output/doh_dataset --data_dir ../csv_output/doh_dataset
```
```shell
python main.py --explain --model_type CNN --explain_checkpoint_path CNN_best_model_checkpoint.pth --explain_data_dir ../csv_output/doh_dataset --data_dir ../csv_output/doh_dataset
```


```shell
python main.py --explain --model_type RandomForest --explain_checkpoint_path RandomForest_best_model_checkpoint.pkl --explain_data_dir ../csv_output/doh_dataset --data_dir ../csv_output/doh_dataset
```
```shell
python main.py --explain --model_type XGBoost --explain_checkpoint_path XGBoost_best_model_checkpoint.pkl --explain_data_dir ../csv_output/doh_dataset --data_dir ../csv_output/doh_dataset
```

---
## Workflow

### Pre-Training
1. Train the model on the primary dataset.
2. Save the best-performing checkpoint for future fine-tuning or testing.

#### Command Example:
```shell
python main.py --model_type CNN_BiLSTM_Attention --train --epochs 10 --batch_size 256 --data_dir ../csv_output/doh_dataset
```
---

### Fine-Tuning
1. Load the pre-trained checkpoint.
2. Train the model on a new, related dataset to adapt it to specific tasks.
   
#### Command Example:
```shell
python main.py --model_type BiLSTM --fine_tune --fine_tune_data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 --fine_tune_epochs 10 --best_checkpoint_path BiLSTM_best_model_checkpoint.pth
```
---

### Testing
1. Use the fine-tuned or pre-trained model to evaluate its performance on the test set.
   
#### Command Example:
```shell
python main.py --model_type XGBoost --test --test_data_dir ../csv_output/CIRA-CIC-DoHBrw-2020 --best_checkpoint_path XGBoost_fine_tuned_model.pkl
```
---

## Notes

### Environment Setup
- Ensure your environment is configured with the correct dependencies and library versions.
- Verify your CUDA installation to match the required configurations for maximum GPU utilization.
- Follow the [PyTorch Installation Guide](https://pytorch.org/get-started/previous-versions/) for step-by-step instructions to set up PyTorch with the appropriate CUDA version.

### Support and Contributions
We greatly value contributions and feedback from the community. If you would like to contribute to this project or encounter any issues:
- Open an issue on this repository to report bugs or suggest new features.
- For further questions or collaboration inquiries, feel free to contact us at:  
📧 **[INSERT EMAIL HERE]**

### Citation
If you find this codebase helpful in your research or work, please consider citing the corresponding paper:
