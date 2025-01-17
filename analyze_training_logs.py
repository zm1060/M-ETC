import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import numpy as np

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class MetricsConfig:
    """度量指标配置"""
    metrics_to_track: List[str] = None
    save_path: str = "metrics_results"
    plot_dpi: int = 300
    
    def __post_init__(self):
        self.metrics_to_track = self.metrics_to_track or [
            "accuracy", "precision", "recall", "f1", "loss"
        ]
        os.makedirs(self.save_path, exist_ok=True)

class MetricsTracker:
    """跟踪和记录训练指标"""
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.history: Dict = {}
        self.summary_stats: Dict = {}
        
    def add_metrics(self, model: str, dataset: str, phase: str, epoch_metrics: Dict):
        """添加一个epoch的指标"""
        if model not in self.history:
            self.history[model] = {}
            logging.info(f"Created new entry for model: {model}")
            
        if dataset not in self.history[model]:
            self.history[model][dataset] = {}
            logging.info(f"Created new entry for dataset: {dataset}")
            
        if phase not in self.history[model][dataset]:
            self.history[model][dataset][phase] = []
            logging.info(f"Created new entry for phase: {phase}")
            
        self.history[model][dataset][phase].append(epoch_metrics)
        logging.debug(f"Added metrics for {model}/{dataset}/{phase} - Epoch {epoch_metrics['epoch']}")
        
    def compute_summary_statistics(self):
        """计算汇总统计信息"""
        for model in self.history:
            self.summary_stats[model] = {}
            for dataset in self.history[model]:
                self.summary_stats[model][dataset] = {}
                for phase in self.history[model][dataset]:
                    metrics_df = pd.DataFrame(self.history[model][dataset][phase])
                    self.summary_stats[model][dataset][phase] = {
                        metric: {
                            'max': metrics_df[metric].max(),
                            'min': metrics_df[metric].min(),
                            'mean': metrics_df[metric].mean(),
                            'std': metrics_df[metric].std()
                        } for metric in self.config.metrics_to_track
                    }
    
    def save_results(self):
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存原始数据
        with open(f"{self.config.save_path}/metrics_history_{timestamp}.json", 'w') as f:
            json.dump(self.history, f, indent=4)
            
        # 保存统计摘要
        with open(f"{self.config.save_path}/metrics_summary_{timestamp}.json", 'w') as f:
            json.dump(self.summary_stats, f, indent=4)

    def compute_cross_validation_stats(self):
        """计算交叉验证的统计信息"""
        for model in self.history:
            self.summary_stats[model] = {}
            for dataset in self.history[model]:
                self.summary_stats[model][dataset] = {}
                for phase in ['train', 'validation']:
                    if phase not in self.history[model][dataset]:
                        continue
                        
                    metrics_df = pd.DataFrame(self.history[model][dataset][phase])
                    
                    # 按fold分组计算最佳性能
                    best_per_fold = metrics_df.groupby('fold').agg({
                        'accuracy': 'max',
                        'precision': 'max',
                        'recall': 'max',
                        'f1': 'max',
                        'loss': 'min'
                    })
                    
                    # 计算交叉验证统计
                    self.summary_stats[model][dataset][phase] = {
                        metric: {
                            'mean': best_per_fold[metric].mean(),
                            'std': best_per_fold[metric].std(),
                            'max': best_per_fold[metric].max(),
                            'min': best_per_fold[metric].min(),
                            'per_fold': best_per_fold[metric].to_dict()
                        }
                        for metric in self.config.metrics_to_track
                    }

class LogParser:
    """日志解析器"""
    def __init__(self):
        self.model_switch_pattern = re.compile(r"root - INFO - Logger initialized for model: (\S+)")
        self.dataset_switch_pattern = re.compile(r"root - INFO - Training data directory: ([\./\w-]+)")
        
        # 深度学习模型的指标模式
        self.dl_metrics_pattern = re.compile(
            r"root - INFO - Epoch (\d+)/\d+, (Train|Validation) Metrics: ({.*?})",
            re.DOTALL
        )
        
        # 传统机器学习模型的指标模式 - 更新以匹配完整的字典
        self.ml_metrics_pattern = re.compile(
            r"root - INFO - ({.*?'train'.*?'confusion_matrix': \[\[.*?\]\].*?'val'.*?'confusion_matrix': \[\[.*?\]\].*?})",
            re.DOTALL
        )
        
    def parse_file(self, file_path: str, metrics_tracker: MetricsTracker) -> None:
        try:
            with open(file_path, 'r') as log_file:
                content = log_file.read()
                
            # 识别模型
            model_match = self.model_switch_pattern.search(content)
            if not model_match:
                logging.error("No model name found in log file")
                return
            current_model = model_match.group(1)
            logging.info(f"Found model: {current_model}")
            
            # 识别数据集
            dataset_match = self.dataset_switch_pattern.search(content)
            if not dataset_match:
                logging.error("No dataset name found in log file")
                return
            dataset_path = dataset_match.group(1)
            current_dataset = dataset_path.rstrip('/').split('/')[-1]
            logging.info(f"Found dataset: {current_dataset}")
            
            # 根据模型类型选择解析方法
            if self._is_deep_learning_model(current_model):
                self._parse_dl_metrics(content, current_model, current_dataset, metrics_tracker)
            else:
                self._parse_ml_metrics(content, current_model, current_dataset, metrics_tracker)
                
        except Exception as e:
            logging.error(f"Error parsing file {file_path}: {str(e)}")
            raise
            
    def _is_deep_learning_model(self, model_name: str) -> bool:
        """判断是否为深度学习模型"""
        dl_models = {'MLP','CNN', 'RNN', 'LSTM', 'GRU', 'Attention', 'BiGRU', 'BiLSTM'}
        return any(name in model_name for name in dl_models)
    
    def _parse_dl_metrics(self, content: str, model: str, dataset: str, metrics_tracker: MetricsTracker):
        """解析深度学习模型的指标"""
        current_fold = 1
        fold_pattern = re.compile(r"fold_(\d+)_checkpoint")
        
        for line in content.split('\n'):
            # 检查fold切换
            fold_match = fold_pattern.search(line)
            if fold_match:
                current_fold = int(fold_match.group(1))
                continue
                
            # 解析训练/验证指标
            metrics_match = self.dl_metrics_pattern.search(line)
            if metrics_match:
                try:
                    epoch = int(metrics_match.group(1))
                    phase = metrics_match.group(2).lower()
                    metrics_dict = eval(metrics_match.group(3))
                    
                    metrics_dict['epoch'] = epoch
                    metrics_dict['fold'] = current_fold
                    
                    metrics_tracker.add_metrics(
                        model, dataset, phase, metrics_dict
                    )
                    logging.debug(f"Added {phase} metrics for {model} fold {current_fold} epoch {epoch}")
                    
                except Exception as e:
                    logging.error(f"Error processing DL metrics: {str(e)}")
                    continue
    
    def _parse_ml_metrics(self, content: str, model: str, dataset: str, metrics_tracker: MetricsTracker):
        """解析传统机器学习模型的指标"""
        current_fold = 1
        
        # 按行分割内容
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "'train'" in line and "'val'" in line:
                try:
                    # 提取完整的指标字符串
                    metrics_str = line[line.find('{'):].strip()
                    if not metrics_str.endswith('}'):
                        continue
                        
                    metrics_dict = eval(metrics_str)
                    
                    # 处理训练指标
                    if 'train' in metrics_dict:
                        train_metrics = metrics_dict['train']
                        train_metrics['epoch'] = 1
                        train_metrics['fold'] = current_fold
                        metrics_tracker.add_metrics(
                            model, dataset, 'train', train_metrics
                        )
                        logging.debug(f"Added train metrics for {model} fold {current_fold}")
                    
                    # 处理验证指标
                    if 'val' in metrics_dict:
                        val_metrics = metrics_dict['val']
                        val_metrics['epoch'] = 1
                        val_metrics['fold'] = current_fold
                        metrics_tracker.add_metrics(
                            model, dataset, 'validation', val_metrics
                        )
                        logging.debug(f"Added validation metrics for {model} fold {current_fold}")
                    
                    current_fold += 1
                    
                except Exception as e:
                    logging.error(f"Error processing ML metrics in fold {current_fold}: {str(e)}")
                    logging.debug(f"Problematic line: {line[:200]}...")
                    continue

class MetricsPlotter:
    """指标可视化器"""
    def __init__(self, config: MetricsConfig):
        self.config = config
        
    def plot_metrics(self, data: Dict, model: str, dataset: str) -> None:
        """为单个模型绘制训练过程图表"""
        if model not in data or dataset not in data[model]:
            return
            
        model_data = data[model][dataset]
        folds = self._get_unique_folds(model_data)
        
        # 判断是否为深度学习模型
        is_dl_model = any(name in model for name in {'MLP', 'CNN', 'RNN', 'LSTM', 'GRU', 'Attention', 'BiGRU', 'BiLSTM'})
        
        if is_dl_model:
            # 深度学习模型：绘制训练过程折线图
            for fold in folds:
                fig = plt.figure(figsize=(15, 10))
                
                for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1'], 1):
                    plt.subplot(2, 2, i)
                    self._plot_metric_for_fold(model_data, metric, fold)
                
                plt.tight_layout()
                plot_filename = os.path.join(
                    self.config.save_path,
                    f"{model}_{dataset}_fold_{fold}_metrics.jpg"
                )
                plt.savefig(plot_filename, format='jpg', dpi=self.config.plot_dpi)
                plt.close()
        else:
            # 传统机器学习模型：绘制每个fold的柱状图
            fig = plt.figure(figsize=(15, 10))
            
            for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1'], 1):
                plt.subplot(2, 2, i)
                self._plot_ml_metric_summary(model_data, metric)
            
            # plt.suptitle(f"{model} Performance Across Folds", fontsize=14, y=1.02)
            plt.tight_layout()
            plot_filename = os.path.join(
                self.config.save_path,
                f"{model}_{dataset}_summary.jpg"
            )
            plt.savefig(plot_filename, format='jpg', dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()

    def _plot_ml_metric_summary(self, model_data: Dict, metric: str) -> None:
        """为传统机器学习模型绘制指标汇总图"""
        train_data = []
        val_data = []
        folds = []
        
        # 收集每个fold的数据
        for phase in ['train', 'validation']:
            if phase not in model_data:
                continue
            
            for entry in model_data[phase]:
                if phase == 'train':
                    train_data.append(entry[metric])
                else:
                    val_data.append(entry[metric])
                if phase == 'train':  # 只需要添加一次
                    folds.append(f"Fold {entry['fold']}")
        
        # 设置柱状图
        x = np.arange(len(folds))
        width = 0.35
        
        plt.bar(x - width/2, train_data, width, label='Train', color='lightcoral')
        plt.bar(x + width/2, val_data, width, label='Validation', color='lightblue')
        
        # plt.title(f'{metric.title()} Across Folds')
        plt.xlabel('Fold')
        plt.ylabel(metric.title())
        plt.xticks(x, folds)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for i, v in enumerate(train_data):
            plt.text(i - width/2, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
        for i, v in enumerate(val_data):
            plt.text(i + width/2, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    def plot_cross_validation_comparison(self, data: Dict, metric: str, summary_stats: Dict) -> None:
        """绘制交叉验证结果对比图"""
        comparison_data = []
        
        # 获取所有模型并分类
        dl_models = []
        ml_models = []
        
        for model in data.keys():
            if any(name in model for name in {'MLP', 'CNN', 'RNN', 'LSTM', 'GRU', 'Attention', 'BiGRU', 'BiLSTM'}):
                dl_models.append(model)
            else:
                ml_models.append(model)
        
        # 按字母顺序排序
        dl_models.sort()
        ml_models.sort()
        
        # 使用更多不同的鲜艳颜色
        distinct_colors = [
            '#FF0000',  # 红色
            '#00FF00',  # 绿色
            '#0000FF',  # 蓝色
            '#FFD700',  # 金色
            '#FF1493',  # 深粉色
            '#00FFFF',  # 青色
            '#FF8C00',  # 深橙色
            '#8A2BE2',  # 紫罗兰色
            '#32CD32',  # 酸橙色
            '#FF69B4',  # 粉红色
            '#4169E1',  # 皇家蓝
            '#8B4513',  # 马鞍棕色
            '#2E8B57',  # 海洋绿
            '#9370DB',  # 中紫色
            '#B8860B',  # 暗金色
            '#4B0082',  # 靛青色
            '#20B2AA',  # 浅海洋绿
            '#FF6347',  # 番茄色
            '#7B68EE',  # 中暗蓝色
            '#3CB371',  # 中海洋绿
            '#DC143C',  # 猩红色
            '#00FA9A',  # 中春绿色
            '#9932CC',  # 暗兰花色
            '#8B0000',  # 暗红色
            '#006400',  # 暗绿色
            '#483D8B',  # 暗板岩蓝
            '#FF4500',  # 橙红色
            '#DA70D6',  # 兰花色
            '#556B2F',  # 暗橄榄绿
            '#8B008B',  # 暗洋红
            '#2F4F4F',  # 暗岩灰
            '#FF8C69',  # 鲑鱼色
            '#4682B4',  # 钢蓝色
            '#9400D3',  # 暗紫色
            '#CD853F',  # 秘鲁色
            '#708090',  # 岩灰色
            '#00CED1',  # 暗宝石绿
            '#8B4726',  # 马鞍棕色
            '#F4A460',  # 沙褐色
            '#D2691E',  # 巧克力色
        ]
        
        # 为所有模型分配颜色
        colors = {
            model: distinct_colors[i % len(distinct_colors)]
            for i, model in enumerate(dl_models + ml_models)
        }
        
        # 收集数据（先深度学习模型，后机器学习模型）
        for model in dl_models + ml_models:
            for dataset in data[model]:
                if 'validation' in data[model][dataset]:
                    stats = summary_stats[model][dataset]['validation'][metric]
                    comparison_data.append({
                        'Model': model,
                        'Type': 'Deep Learning' if model in dl_models else 'Machine Learning',
                        'Dataset': dataset,
                        'Mean': stats['mean'],
                        'Std': stats['std']
                    })
        
        if not comparison_data:
            logging.warning(f"No comparison data available for metric: {metric}")
            return
        
        df = pd.DataFrame(comparison_data)
        
        plt.figure(figsize=(14, 6))
        
        # 绘制带颜色的柱状图
        ax = sns.barplot(
            x='Model', 
            y='Mean', 
            data=df,
            palette=[colors[model] for model in df['Model']]
        )
        
        # 添加分隔线
        if dl_models and ml_models:
            plt.axvline(x=len(dl_models)-0.5, color='gray', linestyle='--', alpha=0.5)
        
        # 添加误差条
        for i, row in df.iterrows():
            ax.errorbar(
                i, row['Mean'], 
                yerr=row['Std'], 
                color='black', 
                capsize=5,
                capthick=1,
                elinewidth=1
            )
        
        # 设置图表样式
        # plt.title(f'Cross-validation {metric.title()} Comparison', fontsize=12, pad=15)
        plt.ylabel(f'{metric.title()} Score', fontsize=10)
        plt.xlabel('Model', fontsize=10)
        plt.xticks(rotation=90, ha='right')
        
        # 添加数值标签
        for i, v in enumerate(df['Mean']):
            ax.text(
                i, v + df['Std'].iloc[i], 
                f'{v:.4f}', 
                ha='center', 
                va='bottom',
                fontsize=8
            )
        
        # 添加模型类型标注
        if dl_models:
            plt.text(len(dl_models)/2 - 0.5, plt.ylim()[1], 'Deep Learning Models',
                    ha='center', va='bottom', fontsize=10)
        if ml_models:
            plt.text(len(dl_models) + len(ml_models)/2 - 0.5, plt.ylim()[1], 'Machine Learning Models',
                    ha='center', va='bottom', fontsize=10)
        
        # 设置y轴范围，留出空间显示误差条和标签
        ymin, ymax = plt.ylim()
        plt.ylim(ymin, ymax + (ymax - ymin) * 0.15)  # 增加顶部空间以容纳标签
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图表
        plot_filename = os.path.join(
            self.config.save_path,
            f"cv_comparison_{metric}.jpg"
        )
        plt.savefig(plot_filename, format='jpg', dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
    def _get_unique_folds(self, model_data: Dict) -> List[int]:
        """获取所有唯一的fold编号"""
        folds = set()
        for phase in ['train', 'validation']:
            if phase in model_data:
                for entry in model_data[phase]:
                    folds.add(entry['fold'])
        return sorted(list(folds))
        
    def _plot_metric_for_fold(self, model_data: Dict, metric: str, fold: int) -> None:
        """为特定fold绘制指标图"""
        plt.figure(figsize=(10, 6))  # 确保每个指标有自己的图
        
        for phase in ['train', 'validation']:
            if phase not in model_data:
                continue
                
            # 过滤当前fold的数据并按epoch排序
            fold_data = [
                entry for entry in model_data[phase] 
                if entry['fold'] == fold
            ]
            
            if not fold_data:
                continue
                
            # 创建DataFrame并按epoch排序
            df = pd.DataFrame(fold_data)
            df = df.sort_values('epoch')  # 添加这行来排序
            
            # 绘制曲线
            plt.plot(df['epoch'], df[metric], 
                    label=f'{phase.capitalize()}',
                    marker='o' if phase == 'validation' else None,
                    linestyle='-',  # 添加显式的线型
                    markersize=4)   # 调整标记大小
        
        # plt.title(f'{metric.title()} (Fold {fold})')
        plt.xlabel('Epoch')
        plt.ylabel(metric.title())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

def main():
    # 设置更详细的日志记录
    logging.getLogger().setLevel(logging.DEBUG)
    
    # 初始化配置
    config = MetricsConfig(
        save_path="analysis_results",
        plot_dpi=300
    )
    
    # 初始化组件
    metrics_tracker = MetricsTracker(config)
    log_parser = LogParser()
    metrics_plotter = MetricsPlotter(config)
    
    # 自动获取logs目录中的所有日志文件
    log_dir = './logs'
    if not os.path.exists(log_dir):
        logging.error(f"Logs directory not found: {log_dir}")
        return
        
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    if not log_files:
        logging.error("No log files found in logs directory")
        return
        
    logging.info(f"Found {len(log_files)} log files: {', '.join(log_files)}")
    
    # 解析所有日志文件
    for log_file in log_files:
        try:
            log_path = os.path.join(log_dir, log_file)
            logging.info(f"Processing log file: {log_file}")
            
            # 解析日志
            log_parser.parse_file(log_path, metrics_tracker)
            
        except Exception as e:
            logging.error(f"Error processing {log_file}: {str(e)}")
            continue
    
    # 验证是否有数据
    if not metrics_tracker.history:
        logging.error("No metrics were collected!")
        return
        
    # 计算统计信息并保存
    metrics_tracker.compute_cross_validation_stats()
    metrics_tracker.save_results()
    
    # 打印数据摘要
    for model in metrics_tracker.history:
        for dataset in metrics_tracker.history[model]:
            for phase in metrics_tracker.history[model][dataset]:
                num_entries = len(metrics_tracker.history[model][dataset][phase])
                logging.info(f"Collected {num_entries} entries for {model}/{dataset}/{phase}")
    
    # 生成单个模型的详细图表
    for model in metrics_tracker.history:
        for dataset in metrics_tracker.history[model]:
            metrics_plotter.plot_metrics(
                metrics_tracker.history, model, dataset
            )
    
    # 生成交叉验证对比图表
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'loss']:
        metrics_plotter.plot_cross_validation_comparison(
            metrics_tracker.history, 
            metric,
            metrics_tracker.summary_stats
        )

if __name__ == "__main__":
    main()
