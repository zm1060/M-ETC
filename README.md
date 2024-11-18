# Project README

## Announcement:
This project is the official implementation of the code for the paper titled "INSERT PAPER TITLE HERE". If you use this code or find it helpful in your research, please cite the paper:
xxx


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

### Training
Train a model using the specified architecture and dataset.

#### Example Commands
```shell
python main.py --model_type BiLSTM --train --epochs 10 --batch_size 64 --data_dir ./csv_output/doh_dataset
```
```shell
python main.py --model_type CNN_BiLSTM_Attention --train --epochs 10 --batch_size 64 --data_dir ./csv_output/doh_dataset
```
```shell
python main.py --model_type XGBoost --train --epochs 10 --batch_size 64 --data_dir ./csv_output/doh_dataset
```
```shell
python main.py --model_type RandomForest --train --epochs 10 --batch_size 64 --data_dir ./csv_output/doh_dataset
```
---

### Fine-Tuning
Fine-tune a pre-trained model on a new dataset.

#### Example Commands
```shell
python main.py --model_type CNN_BiLSTM_Attention --fine_tune --fine_tune_data_dir ./csv_output/CIRA-CIC-DoHBrw-2020 --fine_tune_epochs 10 --best_checkpoint_path CNN_BiLSTM_Attention_best_model_checkpoint.pth
```
```shell
python main.py --model_type BiLSTM --fine_tune --fine_tune_data_dir ./csv_output/CIRA-CIC-DoHBrw-2020 --fine_tune_epochs 10 --best_checkpoint_path BiLSTM_best_model_checkpoint.pth
````
```shell
python main.py --model_type RandomForest --fine_tune --fine_tune_data_dir ./csv_output/CIRA-CIC-DoHBrw-2020 --fine_tune_epochs 10 --best_checkpoint_path RandomForest_model.pkl
```
```shell
python main.py --model_type XGBoost --fine_tune --fine_tune_data_dir ./csv_output/CIRA-CIC-DoHBrw-2020 --fine_tune_epochs 10 --best_checkpoint_path XGBoost_model.pkl
```
---

### Testing
Evaluate the performance of a trained or fine-tuned model on a test dataset(never used before).

#### Example Commands
```shell
python main.py --model_type CNN_BiLSTM_Attention --test --test_data_dir ./csv_output/CIRA-CIC-DoHBrw-2020 --best_checkpoint_path CNN_BiLSTM_Attention_fine_tuned_model.pth
```
```shell
python main.py --model_type BiLSTM --test --test_data_dir ./csv_output/CIRA-CIC-DoHBrw-2020 --best_checkpoint_path BiLSTM_fine_tuned_model.pth
```
```shell
python main.py --model_type XGBoost --test --test_data_dir ./csv_output/CIRA-CIC-DoHBrw-2020 --best_checkpoint_path XGBoost_fine_tuned_model.pkl
```
```shell
python main.py --model_type RandomForest --test --test_data_dir ./csv_output/CIRA-CIC-DoHBrw-2020 --best_checkpoint_path RandomForest_fine_tuned_model.pkl
```
---

### Explain
Explain

#### Example Commands
```shell
 python main.py --explain --model_type CNN_BiLSTM_Attention --explain_checkpoint_p
ath CNN_BiLSTM_best_model_checkpoint.pth --explain_data_dir ./csv_output/doh_dataset --data_dir ./csv_output
/doh_dataset
```
```shell
 python main.py --explain --model_type CNN_BiLSTM_Attention --explain_checkpoint_p
ath CNN_BiLSTM_best_model_checkpoint.pth --explain_data_dir ./csv_output/doh_dataset --data_dir ./csv_output
/doh_dataset
```
```shell
 python main.py --explain --model_type CNN_BiLSTM_Attention --explain_checkpoint_p
ath CNN_BiLSTM_best_model_checkpoint.pth --explain_data_dir ./csv_output/doh_dataset --data_dir ./csv_output
/doh_dataset
```
```shell
 python main.py --explain --model_type CNN_BiLSTM_Attention --explain_checkpoint_p
ath CNN_BiLSTM_best_model_checkpoint.pth --explain_data_dir ./csv_output/doh_dataset --data_dir ./csv_output
/doh_dataset
```
---
## Workflow

### Pre-Training
1. Train the model on the primary dataset.
2. Save the best-performing checkpoint for future fine-tuning or testing.

#### Command Example:
```shell
python main.py --model_type CNN_BiLSTM_Attention --train --epochs 10 --batch_size 64 --data_dir ./csv_output/doh_dataset
```
---

### Fine-Tuning
1. Load the pre-trained checkpoint.
2. Train the model on a new, related dataset to adapt it to specific tasks.
   
#### Command Example:
```shell
python main.py --model_type BiLSTM --fine_tune --fine_tune_data_dir ./csv_output/CIRA-CIC-DoHBrw-2020 --fine_tune_epochs 10 --best_checkpoint_path BiLSTM_best_model_checkpoint.pth
```
---

### Testing
1. Use the fine-tuned or pre-trained model to evaluate its performance on the test set.
   
#### Command Example:
```shell
python main.py --model_type XGBoost --test --test_data_dir ./csv_output/CIRA-CIC-DoHBrw-2020 --best_checkpoint_path XGBoost_fine_tuned_model.pkl
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
