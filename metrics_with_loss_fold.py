import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_logs_with_fold(file_path):
    """
    Parses a log file to extract training and validation metrics for multiple folds.

    Args:
        file_path (str): Path to the log file.

    Returns:
        dict: Nested dictionary with fold numbers as keys and metrics data as values.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")

    results = {}
    current_fold = None

    # Regular expressions
    fold_pattern = re.compile(r"Model saved after epoch \d+ at .*fold_(\d+)_checkpoint\.pth")
    metrics_pattern = re.compile(
        r"Epoch\s+(\d+)/\d+,\s+(Train|Validation)\s+Metrics:\s+\{.*?'accuracy':\s*([\d.]+),"
        r".*?'precision':\s*([\d.]+),.*?'recall':\s*([\d.]+),.*?'f1':\s*([\d.]+),.*?'loss':\s*([\d.]+)"
    )

    with open(file_path, 'r', encoding='utf-8') as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue

            # Match fold
            fold_match = fold_pattern.search(line)
            if fold_match:
                current_fold = int(fold_match.group(1))
                if current_fold not in results:
                    results[current_fold] = {
                        "train": [],
                        "validation": [],
                    }

            # Match metrics
            metrics_match = metrics_pattern.search(line)
            if metrics_match and current_fold is not None:
                epoch = int(metrics_match.group(1))
                phase = metrics_match.group(2).lower()
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
                    "loss": loss,
                })

    return results

def collect_validation_data(parsed_results):
    """
    将各 fold 的验证集指标整理成一个 DataFrame
    列包括: fold, epoch, accuracy, precision, recall, f1, loss
    """
    all_data = []
    for fold, metrics in parsed_results.items():
        for entry in metrics["validation"]:
            all_data.append({
                "fold": fold,
                "epoch": entry["epoch"],
                "accuracy": entry["accuracy"],
                "precision": entry["precision"],
                "recall": entry["recall"],
                "f1": entry["f1"],
                "loss": entry["loss"],
            })
    df = pd.DataFrame(all_data)
    df.sort_values(by=["epoch"], inplace=True)
    return df

def plot_metrics_with_loss(data, fold, title=None, save_path=None):
    """
    针对一个 Fold，绘制每个指标（Accuracy, Precision, Recall, F1）和 Loss 的对比图，并标注最佳点。
    
    Args:
        data (DataFrame): 包含至少 ['fold', 'epoch', 'accuracy', 'precision', 'recall', 'f1', 'loss'] 列。
        fold (int): 要可视化的 Fold 编号。
        title (str): 图的标题。
        save_path (str): 若不为 None，则保存图像到指定路径，否则直接 plt.show()。
    """
    # 过滤该 fold 的数据
    fold_data = data[data['fold'] == fold]
    if fold_data.empty:
        print(f"No data found for fold {fold}")
        return
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))  # 2行2列的子图布局
    metrics_info = [
        ("accuracy", "Accuracy", axes[0, 0]),
        ("precision", "Precision", axes[0, 1]),
        ("recall", "Recall", axes[1, 0]),
        ("f1", "F1-Score", axes[1, 1]),
    ]
    
    # 绘制每个指标与 loss 的对比图
    for metric_name, metric_label, ax in metrics_info:
        ax.plot(
            fold_data["epoch"],
            fold_data[metric_name],
            marker="o",
            linestyle="-",
            label=f"{metric_label}",
            color="tab:blue",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_label, color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(True)

        # 添加 loss 曲线 (共享右侧坐标轴)
        ax_loss = ax.twinx()
        ax_loss.plot(
            fold_data["epoch"],
            fold_data["loss"],
            marker="x",
            linestyle="--",
            label="Loss",
            color="tab:red",
        )
        ax_loss.set_ylabel("Loss", color="tab:red")
        ax_loss.tick_params(axis="y", labelcolor="tab:red")

        # 找到最佳点
        best_metric_idx = fold_data[metric_name].idxmax()
        best_metric = fold_data.loc[best_metric_idx]
        ax.annotate(
            f"{metric_label}: {best_metric[metric_name]:.4f}",
            (best_metric["epoch"], best_metric[metric_name]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="blue",
            arrowprops=dict(arrowstyle="->", color="blue", lw=0.5)
        )

        # 添加图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_loss.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # 设置整体标题
    fig.suptitle(title if title else f"Validation Metrics with Loss for Fold {fold}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，避免标题被遮挡

    if save_path:
        plt.savefig(save_path, format='jpg', dpi=300, bbox_inches="tight")
        print(f"Plot saved at {save_path}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    # 1) 日志文件路径
    log_file_path = './logs/CNN_BiGRU_Attention.log'  # 请根据实际路径修改

    try:
        # 2) 解析日志得到结果
        parsed_results = parse_logs_with_fold(log_file_path)

        # 3) 将验证集的所有指标合并到 DataFrame
        df = collect_validation_data(parsed_results)

        # 4) 针对每个 fold 进行可视化
        for i in range(1, 6):
            plot_metrics_with_loss(
                data=df,
                fold=i,
                title=f"Validation Metrics with Loss for Fold {i}",
                save_path=f"metrics_with_loss_fold_{i}.jpg"
            )

    except FileNotFoundError as e:
        print(e)
