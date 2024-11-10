# How to use

cuda12.5
torch 2.4.1+cu124

This is for trian:

```shell
python3 main.py --train 
```

This is for explain:

```shell
python3 main.py --explain --best_checkpoint_path ./best_model_checkpoint.pth
```

## 训练预训练模型

```shell
python main.py --train --epochs 100 --data_dir  ./csv_output/CIRA-CIC-DoHBrw-2020 --checkpoint_path pretrained_model.pth
```

## 在新数据集上微调模型

```shell
python main.py --fine_tune --fine_tune_data_dir ./csv_output/doh_dataset --fine_tune_epochs 10 --checkpoint_path fine_tuned_model.pth --sample_size 0.1
```

```shell
python main.py --fine_tune --fine_tune_data_dir ./csv_output/doh_dataset --fine_tune_epochs 10 --checkpoint_path fine_tuned_model.pth --sample_size 100
```

## 测试微调后的模型性能

```shell
python main.py --test --test_data_dir ./csv_output/doh_dataset --best_checkpoint_path fine_tuned_model.pth
```

```shell
python main.py --test --test_data_dir ./csv_output/doh_dataset --best_checkpoint_path fine_tuned_model.pth
```

## This is for expirements

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

```shell
python main.py --model_type RandomForest --test --test_data_dir ./csv_output/CIRA-CIC-DoHBrw-2020
```
