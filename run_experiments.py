import subprocess
import os
from datetime import datetime

def run_experiment(model_type, data_dir, epochs=100, batch_size=256):
    """运行单个实验"""
    cmd = [
        "python", "main.py",
        "--model_type", model_type,
        "--data_dir", data_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--train",
        "--save_current"
    ]
    subprocess.run(cmd)

def run_ablation_study():
    """进行消融实验"""
    # 定义不同组件组合的模型
    models = [
        'CNN_BiGRU_Attention', # 完整模型
        'CNN_BiGRU',           # 无注意力机制
        'BiGRU_Attention',     # 无CNN
        'CNN',                 # 仅CNN
        'CNN_Attention'        # 无序列处理
    ]
    
    # 定义实验数据集
    datasets = [
        './csv_output/CIRA-CIC-DoHBrw-2020',
        './csv_output/doh_dataset', 
        './csv_output/FiveWeek',
        './csv_output/Generated',
        './csv_output/RealWorld',
        './csv_output/Custom_dataset',
        './csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel',
        './csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD',
        './csv_output/Tunnel/DoH-Tunnel-Traffic-HKD',
    ]
    
    # 在每个数据集上运行所有模型
    for dataset in datasets:
        print(f"\nRunning experiments on dataset: {dataset}")
        for model in models:
            print(f"\nTesting model: {model}")
            run_experiment(model, dataset)

def run_comparison_study():
    """进行对比实验"""
    # 定义待比较的模型
    models = [
        # 传统机器学习模型
        'RandomForest', 'XGBoost', 'LogisticRegression', 'AdaBoost',
        'DecisionTree', 'NaiveBayes', 'LDA', 'ExtraTrees', 'CatBoost', 'LightGBM',
        
        # 深度学习模型
        'MLP', 'DNN', 'RNN', 'GRU', 'LSTM', 'BiGRU', 'BiLSTM', 'CNN',
        'CNN_GRU', 'CNN_LSTM', 'CNN_BiGRU', 'CNN_BiLSTM',
        'CNN_LSTM_Attention', 'CNN_GRU_Attention',
        'CNN_BiGRU_Attention', 'CNN_BiLSTM_Attention'
    ]
    
    # 定义实验数据集
    datasets = [
        './csv_output/CIRA-CIC-DoHBrw-2020',
        './csv_output/doh_dataset',
        './csv_output/FiveWeek',
        './csv_output/Generated',
        './csv_output/RealWorld',
        './csv_output/Custom_dataset',
        './csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel',
        './csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD',
        './csv_output/Tunnel/DoH-Tunnel-Traffic-HKD',
    ]
    
    # 在每个数据集上运行所有模型
    for dataset in datasets:
        print(f"\nRunning experiments on dataset: {dataset}")
        for model in models:
            print(f"\nTesting model: {model}")
            run_experiment(model, dataset)

def run_cross_dataset_study():
    """进行跨数据集实验"""
    # 定义源数据集和目标数据集
    source_dataset = './csv_output/CIRA-CIC-DoHBrw-2020'
    target_datasets = [
        './csv_output/doh_dataset',
        './csv_output/FiveWeek',
        './csv_output/Custom_dataset'
    ]
    
    model = 'CNN_BiGRU_Attention'  # 使用最佳模型
    
    # 首先在源数据集上训练
    run_experiment(model, source_dataset)
    
    # 然后在目标数据集上进行微调和测试
    for target in target_datasets:
        cmd = [
            "python", "main.py",
            "--model_type", model,
            "--data_dir", target,
            "--fine_tune",
            "--fine_tune_epochs", "10",
            "--test"
        ]
        subprocess.run(cmd)

if __name__ == "__main__":
    # 创建实验结果目录
    os.makedirs("experiment_results", exist_ok=True)
    
    # 记录实验开始时间
    start_time = datetime.now()
    print(f"Starting experiments at {start_time}")
    
    # 运行主要实验
    print("Running ablation study...")
    run_ablation_study()
    
    print("\nRunning comparison study...")
    run_comparison_study()
    
    print("\nRunning cross-dataset study...")
    run_cross_dataset_study()
    
    # 记录实验结束时间和总耗时
    end_time = datetime.now()
    print(f"\nExperiments completed at {end_time}")
    print(f"Total time taken: {end_time - start_time}")
    
    # 运行隧道检测实验
    tunnel_datasets = [
        './csv_output/Tunnel/CIRA-CIC-DoHBrw-2020-Tunnel',
        './csv_output/Tunnel/DoH-DGA-Malware-Traffic-HKD', 
        './csv_output/Tunnel/DoH-Tunnel-Traffic-HKD'
    ]
    
    print("\nRunning tunnel detection experiments...")
    for dataset in tunnel_datasets:
        # 训练阶段 - 使用CNN_BiLSTM_Attention模型
        cmd = [
            "python", "main.py",
            "--model_type", "CNN_BiLSTM_Attention",
            "--train",
            "--epochs", "100",
            "--batch_size", "256",
            "--data_dir", dataset
        ]
        subprocess.run(cmd)
        
        # 微调阶段 - 使用BiLSTM模型
        cmd = [
            "python", "main.py", 
            "--model_type", "BiLSTM",
            "--fine_tune",
            "--fine_tune_epochs", "10",
            "--data_dir", dataset,
            "--best_checkpoint_path", "BiLSTM_best_model_checkpoint.pth"
        ]
        subprocess.run(cmd)
        
        # 测试阶段 - 使用传统机器学习模型
        for model in ["RandomForest", "XGBoost"]:
            cmd = [
                "python", "main.py",
                "--model_type", model,
                "--test",
                "--data_dir", dataset,
                "--test_checkpoint_path", f"{model}_fine_tuned_model.pkl"
            ]
            subprocess.run(cmd)
            
        # 可解释性分析阶段
        cmd = [
            "python", "main.py",
            "--explain",
            "--model_type", "CNN_BiLSTM_Attention",
            "--data_dir", dataset,
            "--explain_checkpoint_path", "CNN_BiLSTM_Attention_best_model_checkpoint.pth"
        ]
        subprocess.run(cmd)