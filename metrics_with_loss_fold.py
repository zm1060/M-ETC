import os
import re
import pandas as pd
import numpy as np
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
        r".*?'precision':\s*([\d.]+),.*?'recall':\s*([\d.]+),.*?'f1':\s*([\d.]+),"
        r".*?'confusion_matrix':\s*\[(\[.*?\])\],.*?'loss':\s*([\d.]+)"
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
                confusion_matrix_str = metrics_match.group(7)
                loss = float(metrics_match.group(8))

                # Parse confusion matrix
                confusion_matrix = eval(confusion_matrix_str)

                results[current_fold][phase].append({
                    "epoch": epoch,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "loss": loss,
                    "confusion_matrix": confusion_matrix
                })

    return results

def collect_validation_data(parsed_results):
    """
    将各 fold 的验证集指标整理成一个 DataFrame
    列包括: fold, epoch, accuracy, precision, recall, f1, loss, confusion_matrix
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
                "confusion_matrix": entry["confusion_matrix"]
            })
    df = pd.DataFrame(all_data)
    df.sort_values(by=["epoch"], inplace=True)
    return df

def calculate_roc_points(confusion_matrix):
    """
    根据混淆矩阵计算ROC曲线的点
    
    Args:
        confusion_matrix: 2x2混淆矩阵 [[TN, FP], [FN, TP]]
        
    Returns:
        fpr_list: FPR值列表
        tpr_list: TPR值列表
    """
    TN, FP = confusion_matrix[0]
    FN, TP = confusion_matrix[1]
    
    # 计算TPR和FPR
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    # 为了画出完整的ROC曲线，我们需要添加起点(0,0)和终点(1,1)
    fpr_list = [0, fpr, 1]
    tpr_list = [0, tpr, 1]
    
    return fpr_list, tpr_list

def calculate_pr_points(confusion_matrix):
    """
    根据混淆矩阵计算PR曲线的点
    
    Args:
        confusion_matrix: 2x2混淆矩阵 [[TN, FP], [FN, TP]]
        
    Returns:
        precision_list: Precision值列表
        recall_list: Recall值列表
    """
    TN, FP = confusion_matrix[0]
    FN, TP = confusion_matrix[1]
    
    # 计算Precision和Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0  # 当TP+FP=0时，precision定义为1
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # PR曲线的起点和终点
    # 注意：PR曲线的起点不是(0,0)，而是(0,1)
    precision_list = [1, precision, precision]
    recall_list = [0, recall, 1]
    
    return precision_list, recall_list

def plot_roc_curves(data, fold=None, title=None, save_path=None):
    """
    绘制ROC曲线。
    
    Args:
        data (DataFrame): 包含混淆矩阵的数据。
        fold (int): 要可视化的Fold编号，如果为None则绘制所有fold。
        title (str): 图的标题。
        save_path (str): 保存图像的路径。
    """
    # 创建两个子图：一个显示完整ROC曲线，一个显示放大的顶部区域
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    if fold is not None:
        fold_data = data[data['fold'] == fold]
        if fold_data.empty:
            print(f"No data found for fold {fold}")
            return
        folds_to_plot = [fold]
    else:
        folds_to_plot = data['fold'].unique()
    
    # 用于存储所有fold的最小FPR和最大TPR，以确定放大区域的范围
    min_fpr, max_tpr = 1.0, 0.0
    
    for current_fold in folds_to_plot:
        fold_data = data[data['fold'] == current_fold]
        
        # 获取最后一个epoch的数据
        last_epoch_data = fold_data.iloc[-1]
        confusion_matrix = last_epoch_data['confusion_matrix']
        
        # 计算ROC曲线的点
        fpr, tpr = calculate_roc_points(confusion_matrix)
        
        # 更新最小FPR和最大TPR
        min_fpr = min(min_fpr, min(fpr))
        max_tpr = max(max_tpr, max(tpr))
        
        # 计算AUC
        auc_score = np.trapz(tpr, fpr)  # 使用梯形法则计算AUC
        
        # 在两个子图中都绘制ROC曲线
        for ax in [ax1, ax2]:
            ax.plot(fpr, tpr, lw=2,
                   label=f'Fold {current_fold} (AUC = {auc_score:.3f})')
    
    # 在两个子图中都绘制对角线
    for ax in [ax1, ax2]:
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.grid(True)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
    
    # 设置完整ROC曲线的范围
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_title('Complete ROC Curves')
    ax1.legend(loc="lower right")
    
    # 设置放大区域的范围
    zoom_x_min = 0.0
    zoom_x_max = 0.2  # 调整这个值可以控制放大区域的宽度
    zoom_y_min = 0.8  # 调整这个值可以控制放大区域的下限
    zoom_y_max = 1.01  # 调整这个值可以控制放大区域的上限
    
    ax2.set_xlim([zoom_x_min, zoom_x_max])
    ax2.set_ylim([zoom_y_min, zoom_y_max])
    ax2.set_title('Zoomed ROC Curves (Top Area)')
    ax2.legend(loc="lower right")
    
    # 设置整体标题
    fig.suptitle(title if title else 'Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, format='jpg', dpi=300, bbox_inches="tight")
        print(f"ROC curve saved at {save_path}")
        plt.close()
    else:
        plt.show()

def plot_pr_curves(data, fold=None, title=None, save_path=None):
    """
    绘制PR曲线。
    
    Args:
        data (DataFrame): 包含混淆矩阵的数据。
        fold (int): 要可视化的Fold编号，如果为None则绘制所有fold。
        title (str): 图的标题。
        save_path (str): 保存图像的路径。
    """
    # 创建两个子图：一个显示完整PR曲线，一个显示放大的顶部区域
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    if fold is not None:
        fold_data = data[data['fold'] == fold]
        if fold_data.empty:
            print(f"No data found for fold {fold}")
            return
        folds_to_plot = [fold]
    else:
        folds_to_plot = data['fold'].unique()
    
    # 用于存储所有fold的最小Recall和最小Precision，以确定放大区域的范围
    min_recall, min_precision = 1.0, 1.0
    
    for current_fold in folds_to_plot:
        fold_data = data[data['fold'] == current_fold]
        
        # 获取最后一个epoch的数据
        last_epoch_data = fold_data.iloc[-1]
        confusion_matrix = last_epoch_data['confusion_matrix']
        
        # 计算PR曲线的点
        precision, recall = calculate_pr_points(confusion_matrix)
        
        # 更新最小值
        min_recall = min(min_recall, min(recall))
        min_precision = min(min_precision, min(precision))
        
        # 计算AP (Average Precision)
        ap_score = np.trapz(precision, recall)  # 使用梯形法则计算AP
        
        # 在两个子图中都绘制PR曲线
        for ax in [ax1, ax2]:
            ax.plot(recall, precision, lw=2,
                   label=f'Fold {current_fold} (AP = {ap_score:.3f})')
    
    # 在两个子图中设置基本属性
    for ax in [ax1, ax2]:
        ax.grid(True)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
    
    # 设置完整PR曲线的范围
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_title('Complete PR Curves')
    ax1.legend(loc="lower left")
    
    # 设置放大区域的范围
    zoom_x_min = 0.8  # 调整这个值可以控制放大区域的起始recall
    zoom_x_max = 1.0  # 调整这个值可以控制放大区域的终止recall
    zoom_y_min = 0.8  # 调整这个值可以控制放大区域的最小precision
    zoom_y_max = 1.01  # 调整这个值可以控制放大区域的最大precision
    
    ax2.set_xlim([zoom_x_min, zoom_x_max])
    ax2.set_ylim([zoom_y_min, zoom_y_max])
    ax2.set_title('Zoomed PR Curves (Top-Right Area)')
    ax2.legend(loc="lower left")
    
    # 设置整体标题
    fig.suptitle(title if title else 'Precision-Recall Curve', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, format='jpg', dpi=300, bbox_inches="tight")
        print(f"PR curve saved at {save_path}")
        plt.close()
    else:
        plt.show()

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
            # 绘制metrics和loss图
            plot_metrics_with_loss(
                data=df,
                fold=i,
                title=f"Validation Metrics with Loss for Fold {i}",
                save_path=f"metrics_with_loss_fold_{i}.jpg"
            )
        
        # 5) 绘制所有fold的ROC曲线在同一张图上
        plot_roc_curves(
            data=df,
            title="ROC Curves for All Folds",
            save_path="roc_curves_all_folds.jpg"
        )
        
        # 6) 绘制所有fold的PR曲线在同一张图上
        plot_pr_curves(
            data=df,
            title="PR Curves for All Folds",
            save_path="pr_curves_all_folds.jpg"
        )

    except FileNotFoundError as e:
        print(e)
