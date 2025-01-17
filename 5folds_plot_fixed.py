
import re
import os
import pandas as pd
import matplotlib.pyplot as plt

def parse_logs_with_fold(file_path):
    """
    解析包含多折训练日志的文件，
    并返回每一折的训练和验证指标。
    
    Args:
        file_path (str): 日志文件路径。
        
    Returns:
        dict: 类似 {
            1: {'train': [...], 'validation': [...], 'train_time': [...], 'inference_time': [...]},
            2: {...},
            ...
        }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")

    results = {}
    current_fold = None

    # 下面这个正则用来捕获“Model saved after epoch X at ...fold_5_checkpoint.pth”
    # 根据你的日志格式修改关键字，这里我假设就是CNN_BiGRU_Attention_fold_X_checkpoint.pth
    fold_pattern = re.compile(r"Model saved after epoch \d+ at .*fold_(\d+)_checkpoint\.pth")

    # 训练时间、推理时间
    train_time_pattern = re.compile(r"Train time:\s+([\d.]+)\s+seconds")
    inference_time_pattern = re.compile(r"Inference time:\s+([\d.]+)\s+seconds")

    # 捕获"Epoch X/Y, Train(或Validation) Metrics: {...}"这样的行
    # 注意：这里写得比较死板，如果日志格式略有不同，需要做相应调整。
    metrics_pattern = re.compile(
        r"Epoch\s+(\d+)\/\d+,\s+(Train|Validation)\s+Metrics:\s+\{.*?'accuracy':\s*([\d.]+),"
        r".*?'precision':\s*([\d.]+),.*?'recall':\s*([\d.]+),.*?'f1':\s*([\d.]+),.*?'loss':\s*([\d.]+)"
    )

    with open(file_path, 'r', encoding='utf-8') as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue

            # 1) 匹配 fold
            fold_match = fold_pattern.search(line)
            if fold_match:
                matched_fold = int(fold_match.group(1))
                current_fold = matched_fold  # 更新current_fold
                # 如果还没有对应的fold，就初始化一下
                if current_fold not in results:
                    results[current_fold] = {
                        "train": [],
                        "validation": [],
                        "train_time": [],
                        "inference_time": []
                    }

            # 2) 捕获训练时间
            train_time_match = train_time_pattern.search(line)
            if train_time_match and current_fold is not None:
                train_time = float(train_time_match.group(1))
                results[current_fold]["train_time"].append(train_time)

            # 3) 捕获推理时间
            inference_time_match = inference_time_pattern.search(line)
            if inference_time_match and current_fold is not None:
                inference_time = float(inference_time_match.group(1))
                results[current_fold]["inference_time"].append(inference_time)

            # 4) 捕获训练/验证指标
            metrics_match = metrics_pattern.search(line)
            if metrics_match and current_fold is not None:
                epoch = int(metrics_match.group(1))
                phase = metrics_match.group(2).lower()  # "train" or "validation"
                accuracy = float(metrics_match.group(3))
                precision = float(metrics_match.group(4))
                recall = float(metrics_match.group(5))
                f1 = float(metrics_match.group(6))
                loss = float(metrics_match.group(7))

                results[current_fold][phase].append({
                    "epoch": epoch,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "loss": loss
                })

    return results


def plot_metrics_folds(data, metric="accuracy", title=None, save_path=None):
    """
    绘制某一指标（accuracy、loss、precision、recall、f1等）在所有fold上的训练/验证曲线，
    并标注表现最好的点。
    """
    plt.figure(figsize=(12, 8))

    epoch_offset = 0  # 起始epoch偏移量

    for fold in sorted(data["fold"].unique()):
        fold_data = data[data["fold"] == fold].copy()
        fold_data["adjusted_epoch"] = fold_data["epoch"] + epoch_offset

        max_adjusted_epoch = fold_data["adjusted_epoch"].max()
        epoch_offset = max_adjusted_epoch  # 更新偏移量

        # 绘制训练曲线
        train_data = fold_data[fold_data["phase"] == "train"]
        plt.plot(
            train_data["adjusted_epoch"],
            train_data[metric],
            label=f"Fold {fold} Train",
            marker="o",
            linestyle="-"
        )

        # 找到训练曲线的最佳点
        if not train_data.empty:
            best_train_idx = train_data[metric].idxmax()
            best_train = train_data.loc[best_train_idx]
            plt.annotate(
                f"{metric.title()}: {best_train[metric]:.4f}",
                (best_train["adjusted_epoch"], best_train[metric]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color="blue"
            )

        # 绘制验证曲线
        val_data = fold_data[fold_data["phase"] == "validation"]
        plt.plot(
            val_data["adjusted_epoch"],
            val_data[metric],
            label=f"Fold {fold} Validation",
            marker="x",
            linestyle="--"
        )

        # 找到验证曲线的最佳点
        if not val_data.empty:
            best_val_idx = val_data[metric].idxmax()
            best_val = val_data.loc[best_val_idx]
            plt.annotate(
                f"{metric.title()}: {best_val[metric]:.4f}",
                (best_val["adjusted_epoch"], best_val[metric]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color="red"
            )

    plt.xlabel("Adjusted Epochs")
    plt.ylabel(metric.title())
    plt.title(title if title else f"{metric.title()} Across Folds")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, format='jpg', dpi=300, bbox_inches="tight")
        print(f"Plot saved at {save_path}")
    else:
        plt.show()



if __name__ == "__main__":
    # 日志文件路径，改成你的实际路径和文件名
    log_file_path = './logs/CNN_BiGRU_Attention.log'
    
    try:
        # 1. 解析日志
        parsed_results = parse_logs_with_fold(log_file_path)

        # 2. 将解析结果转为DataFrame
        all_data = []
        for fold, metrics_dict in parsed_results.items():
            for phase in ["train", "validation"]:
                for entry in metrics_dict[phase]:
                    all_data.append({
                        "fold": fold,
                        "phase": phase,
                        "epoch": entry["epoch"],
                        "accuracy": entry["accuracy"],
                        "precision": entry["precision"],
                        "recall": entry["recall"],
                        "f1": entry["f1"],
                        "loss": entry["loss"],
                    })
        df = pd.DataFrame(all_data)
        df.sort_values(by=["fold", "epoch"], inplace=True)

        # 3. 选择要绘制的指标
        metric_to_plot = "f1"

        # 4. 绘图（并可保存结果）
        plot_metrics_folds(
            data=df,
            metric=metric_to_plot,
            title=f"{metric_to_plot.title()} Across Folds",
            save_path=f"{metric_to_plot}_5folds_plot_fixed.jpg"
        )

    except FileNotFoundError as e:
        print(e)
