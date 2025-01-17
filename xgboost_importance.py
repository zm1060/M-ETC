import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import logging
from typing import Dict, List, Tuple
import json
import re
from data_preprocessing import load_data_from_directory, preprocess_data

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class XGBoostAnalyzer:
    def __init__(self):
        """初始化XGBoost分析器"""
        self.results_dir = 'xgboost_analysis_results'
        os.makedirs(self.results_dir, exist_ok=True)
        self.best_params = self._extract_best_params()
        
    def _extract_best_params(self) -> Dict:
        """从README.md中提取XGBoost的最佳参数"""
        try:
            with open('README.md', 'r') as f:
                content = f.read()
            
            # 使用正则表达式查找XGBoost相关的参数行
            xgb_pattern = r'colsample_bytree=([\d.]+), gamma=([\d.]+), learning_rate=([\d.]+), max_depth=(\d+), n_estimators=(\d+), subsample=([\d.]+)'
            matches = re.findall(xgb_pattern, content)
            
            if not matches:
                logging.warning("No XGBoost parameters found in README.md")
                return self._get_default_params()
            
            # 获取具有最高分数的参数组合
            best_match = matches[0]  # 假设第一个匹配项是最佳参数
            
            return {
                'colsample_bytree': float(best_match[0]),
                'gamma': float(best_match[1]),
                'learning_rate': float(best_match[2]),
                'max_depth': int(best_match[3]),
                'n_estimators': int(best_match[4]),
                'subsample': float(best_match[5])
            }
        except Exception as e:
            logging.error(f"Error extracting parameters: {str(e)}")
            return self._get_default_params()
    
    def _get_default_params(self) -> Dict:
        """返回默认的XGBoost参数"""
        return {
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'learning_rate': 0.001,
            'max_depth': 10,
            'n_estimators': 100,
            'subsample': 0.8
        }

    def _get_datasets(self, base_dir: str = './csv_output') -> List[Tuple[str, str]]:
        """获取所有数据集目录"""
        datasets = []
        
        # 遍历基础目录
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            
            # 如果是目录
            if os.path.isdir(item_path):
                if item == 'Tunnel':
                    # 处理Tunnel子目录
                    for tunnel_dataset in os.listdir(item_path):
                        tunnel_path = os.path.join(item_path, tunnel_dataset)
                        if os.path.isdir(tunnel_path):
                            datasets.append((tunnel_path, f"Tunnel_{tunnel_dataset}"))
                else:
                    datasets.append((item_path, item))
        
        return datasets
    
    def load_dataset(self, dataset_path: str) -> Tuple[pd.DataFrame, pd.Series, List]:
        """使用data_preprocessing.py中的函数加载数据集"""
        # 加载数据
        data = load_data_from_directory(dataset_path)
        
        # 预处理数据
        X, y, _, label_encoder = preprocess_data(data)
        
        return X, y, list(label_encoder.classes_)
    
    def analyze_feature_importance(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
        """使用XGBoost分析特征重要性"""
        # 使用最佳参数初始化XGBoost
        model = XGBClassifier(**self.best_params, random_state=42)
        
        # 训练模型并指定特征名称
        model.fit(X, y)
        
        # 创建特征名称映射 (f0 -> actual_name)
        feature_map = {f'f{i}': name for i, name in enumerate(feature_names)}
        
        # 获取不同类型的特征重要性
        raw_importance = {
            'f_score': dict(zip(feature_names, model.feature_importances_)),
            'gain': model.get_booster().get_score(importance_type='gain'),
            'weight': model.get_booster().get_score(importance_type='weight'),
            'total_gain': model.get_booster().get_score(importance_type='total_gain'),
            'total_cover': model.get_booster().get_score(importance_type='total_cover')
        }
        
        # 将f0, f1等替换为实际的特征名称
        importance_dict = {}
        for imp_type, values in raw_importance.items():
            if imp_type == 'f_score':  # f_score已经使用了正确的特征名称
                importance_dict[imp_type] = values
            else:
                importance_dict[imp_type] = {
                    feature_map[feat]: value 
                    for feat, value in values.items()
                }
        
        return importance_dict
    
    def plot_importance(self, importance_dict: Dict, dataset_name: str, top_n: int = 20):
        """绘制特征重要性图表"""
        importance_types = ['f_score', 'gain', 'weight', 'total_gain', 'total_cover']
        n_plots = len(importance_types)
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 6*n_plots))
        # fig.suptitle(f'XGBoost Feature Importance Analysis - {dataset_name}', fontsize=16, y=0.95)
        
        for ax, imp_type in zip(axes, importance_types):
            if imp_type in importance_dict:
                # 排序并获取前N个特征
                sorted_imp = dict(sorted(
                    importance_dict[imp_type].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_n])
                
                # 创建条形图
                sns.barplot(
                    x=list(sorted_imp.values()),
                    y=list(sorted_imp.keys()),
                    ax=ax
                )
                ax.set_title(f'Top {top_n} Features by {imp_type.replace("_", " ").title()}')
                ax.set_xlabel('Importance')
                ax.set_ylabel('Features')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, f'{dataset_name}_xgboost_importance.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
    
    def save_results(self, importance_dict: Dict, dataset_name: str, classes: List):
        """保存分析结果"""
        # 转换numpy类型为Python原生类型
        def convert_to_native_types(d):
            return {k: float(v) if isinstance(v, (np.number, np.float32, np.float64)) else v 
                   for k, v in d.items()}
        
        results = {
            'parameters': self.best_params,
            'importance': {k: convert_to_native_types(v) for k, v in importance_dict.items()},
            'classes': classes
        }
        
        with open(os.path.join(self.results_dir, f'{dataset_name}_analysis.json'), 'w') as f:
            json.dump(results, f, indent=4)
    
    def analyze_all_datasets(self):
        """分析所有数据集"""
        datasets = self._get_datasets()
        
        for dataset_path, dataset_name in datasets:
            logging.info(f"Analyzing dataset: {dataset_name}")
            try:
                # 加载并分析数据集
                X, y, classes = self.load_dataset(dataset_path)
                
                # 获取特征名称
                data = load_data_from_directory(dataset_path)
                features_to_remove = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp', 'Label', 'label']
                feature_names = [col for col in data.columns if col not in features_to_remove]
                
                # 分析特征重要性
                importance_dict = self.analyze_feature_importance(X, y, feature_names)
                
                # 生成可视化和保存结果
                self.plot_importance(importance_dict, dataset_name)
                self.save_results(importance_dict, dataset_name, classes)
                
                logging.info(f"Completed analysis for {dataset_name}")
            except Exception as e:
                logging.error(f"Error analyzing {dataset_name}: {str(e)}")
                continue

def main():
    analyzer = XGBoostAnalyzer()
    analyzer.analyze_all_datasets()
    
    logging.info("XGBoost feature importance analysis completed")
    logging.info(f"Results saved in {analyzer.results_dir}")

if __name__ == "__main__":
    main() 