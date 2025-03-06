import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import logging
from pathlib import Path
import matplotlib.style as style

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class FeatureAnalyzer:
    def __init__(self):
        self.excluded_columns = ['SourceIP', 'DestinationIP', 'SourcePort', 
                               'DestinationPort', 'TimeStamp']
        self.figure_dpi = 300
        self.correlation_threshold = 0.8
        
    def setup_plotting_style(self):
        """Configure plotting style for better visualization"""
        # Use a clean and modern style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Set custom color palette
        sns.set_palette("husl")
        
        # Configure general plot settings with common fonts
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.linewidth': 1.5,
            'axes.edgecolor': '#333333'
        })
        
    def create_output_directory(self, dataset_dir):
        """创建输出目录"""
        output_dir = Path('analysis_output') / Path(dataset_dir).name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
        
    def plot_feature_importance(self, feature_importance, dataset_name, output_dir):
        """绘制特征重要性图表"""
        plt.figure(figsize=(12, 8))
        
        # 创建渐变色条形图
        colors = sns.color_palette("husl", n_colors=len(feature_importance))
        ax = sns.barplot(x='Importance', y='Feature', data=feature_importance, 
                        palette=colors)
        
        # 添加标题和标签
        # plt.title(f'Feature Importance Analysis - {dataset_name}', fontsize=14, pad=20)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature Name', fontsize=12)
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 美化图表
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.jpg', 
                   dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()
        
    def plot_correlation_matrix(self, correlation_matrix, dataset_name, output_dir):
        """绘制相关性矩阵热力图"""
        # 设置更大的图形尺寸
        plt.figure(figsize=(16, 14))
        
        # 创建蒙版，只显示下三角矩阵
        mask = np.zeros_like(correlation_matrix)
        mask[np.triu_indices_from(mask)] = True
        
        # 设置字体大小和格式化显示
        annot_kws = {
            'size': 8,  # 减小字体大小
        }
        
        # 绘制热力图，优化显示效果
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f',  # 保留两位小数
                   cmap='RdBu_r',  # 使用更好的配色方案
                   center=0,
                   square=True,
                   linewidths=0.8,
                   cbar_kws={"shrink": .8},
                   annot_kws=annot_kws,
                   vmin=-1, vmax=1)  # 固定色彩范围
        
        # 优化标题和布局
        # plt.title(f'Feature Correlation Matrix - {dataset_name}', fontsize=14, pad=20)
        
        # 调整布局，确保图形完整显示
        plt.tight_layout()
        
        # 保存高质量图片
        plt.savefig(output_dir / 'feature_correlation.jpg', 
                   dpi=self.figure_dpi, 
                   bbox_inches='tight',
                   facecolor='white')
        plt.close()

    def plot_combined_feature_importance(self, dataset_results, top_n=10):
        """绘制多个数据集的特征重要性对比图
        
        Args:
            dataset_results: Dict[str, pd.DataFrame] 数据集名称到特征重要性DataFrame的映射
            top_n: int 展示前N个重要特征
        """
        plt.figure(figsize=(15, 8))
        
        # 准备数据
        all_top_features = []
        for dataset_name, importance_df in dataset_results.items():
            # 获取每个数据集的前N个特征
            top_features = importance_df.head(top_n)[['Feature', 'Importance']]
            top_features['Dataset'] = dataset_name
            all_top_features.append(top_features)
        
        # 合并所有数据集的结果
        combined_df = pd.concat(all_top_features, ignore_index=True)
        
        # 创建分组条形图
        ax = sns.barplot(
            x='Feature',
            y='Importance',
            hue='Dataset',
            data=combined_df,
            palette='husl'
        )
        
        # 优化图表样式
        # plt.title(f'Top {top_n} Feature Importance Comparison', fontsize=14, pad=20)
        plt.xlabel('Feature Name', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        
        # 旋转特征标签以防重叠
        plt.xticks(rotation=45, ha='right')
        
        # 添加图例
        plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        output_dir = Path('analysis_output')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(
            output_dir / 'combined_feature_importance.jpg',
            dpi=self.figure_dpi,
            bbox_inches='tight',
            facecolor='white'
        )
        plt.close()

    def analyze_features(self):
        """Execute feature analysis"""
        logger.info("Starting feature analysis...")
        self.setup_plotting_style()
        
        # Get dataset directories (only immediate subdirectories)
        base_dir = Path('../csv_output')
        dataset_dirs = []
        
        # Special handling for Tunnel directory
        tunnel_dir = base_dir / 'Tunnel'
        if tunnel_dir.exists() and tunnel_dir.is_dir():
            tunnel_subdirs = [d for d in tunnel_dir.iterdir() if d.is_dir() and 
                            any(f.suffix == '.csv' for f in d.glob('*.csv'))]
            dataset_dirs.extend(tunnel_subdirs)
            logger.info(f"Found {len(tunnel_subdirs)} datasets in Tunnel directory")
        
        # Get other dataset directories (excluding Tunnel)
        other_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name != 'Tunnel' and 
                     any(f.suffix == '.csv' for f in d.glob('*.csv'))]
        dataset_dirs.extend(other_dirs)
        
        logger.info(f"Found total {len(dataset_dirs)} dataset directories")
        
        if not dataset_dirs:
            logger.warning("No valid dataset directories found. Please ensure that:")
            logger.warning("1. Each directory contains CSV files")
            logger.warning("2. The CSV files contain the required columns")
            return None, None
        
        # Store feature importance results for all datasets
        all_dataset_results = {}
        
        for dataset_dir in tqdm(dataset_dirs, desc="Processing datasets"):
            # Use the directory name directly
            display_name = dataset_dir.name
            logger.info(f"\nAnalyzing dataset: {display_name}")
            
            try:
                # Read and merge CSV files
                csv_files = list(dataset_dir.glob('*.csv'))
                if not csv_files:
                    logger.warning(f"No CSV files found in {display_name}, skipping...")
                    continue
                    
                dfs = []
                for file in tqdm(csv_files, desc=f"Reading CSV files from {display_name}"):
                    try:
                        df = pd.read_csv(file)
                        dfs.append(df)
                    except Exception as e:
                        logger.error(f"Error reading {file.name}: {str(e)}")
                        continue
                
                if not dfs:
                    logger.warning(f"No valid CSV files could be read from {display_name}, skipping...")
                    continue
                
                combined_df = pd.concat(dfs, ignore_index=True)
                logger.info(f"Combined dataset size: {combined_df.shape}")
                
                # Verify required columns exist
                required_columns = self.excluded_columns + ['Label']
                missing_columns = [col for col in required_columns if col not in combined_df.columns]
                if missing_columns:
                    logger.warning(f"Missing required columns in {display_name}: {missing_columns}")
                    continue
                
                # Data preprocessing
                X = combined_df.drop(self.excluded_columns + ['Label'], axis=1)
                le = LabelEncoder()
                y = le.fit_transform(combined_df['Label'])
                
                # Train model
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                model = XGBClassifier(random_state=42)
                model.fit(X_train, y_train)
                
                # Calculate feature importance
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Create output directory and save results
                output_dir = self.create_output_directory(str(dataset_dir))
                
                # Generate visualizations
                self.plot_feature_importance(feature_importance, display_name, output_dir)
                self.plot_correlation_matrix(X.corr(), display_name, output_dir)
                
                # Save analysis results
                feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False)
                X.corr().to_csv(output_dir / 'correlation_matrix.csv')
                
                # Output highly correlated feature pairs
                high_correlation = self.find_high_correlations(X)
                if high_correlation:
                    logger.info("\nHighly correlated feature pairs:")
                    for pair, corr in high_correlation:
                        logger.info(f"{pair}: {corr:.3f}")
                
                # Store current dataset results with display name
                all_dataset_results[display_name] = feature_importance
                
                logger.info(f"\nDataset {display_name} analysis completed!")
                
            except Exception as e:
                logger.error(f"Error processing dataset {display_name}: {str(e)}")
                continue
        
        if not all_dataset_results:
            logger.error("No datasets were successfully processed")
            return None, None
            
        # Generate multi-dataset comparison plot
        logger.info("\nGenerating multi-dataset feature importance comparison...")
        self.plot_combined_feature_importance(all_dataset_results, top_n=10)
        
        logger.info("\nAll dataset analysis completed!")
        return feature_importance, X.corr()
    
    def find_high_correlations(self, X):
        """找出高相关性的特征对"""
        corr_matrix = X.corr()
        high_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= self.correlation_threshold:
                    high_corr.append(
                        ((corr_matrix.columns[i], corr_matrix.columns[j]),
                         corr_matrix.iloc[i, j])
                    )
        
        return sorted(high_corr, key=lambda x: abs(x[1]), reverse=True)

if __name__ == "__main__":
    analyzer = FeatureAnalyzer()
    feature_importance, correlation_matrix = analyzer.analyze_features()
    if feature_importance is not None and correlation_matrix is not None:
        logger.info("Analysis completed successfully!")
    else:
        logger.info("Analysis could not be completed. Please check the warnings above.")