# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# # 主函数：处理所有数据集文件夹
# def process_datasets(main_folder_path):
#     # 遍历主文件夹下的所有子文件夹（每个代表一个数据集）
#     for dataset_name in os.listdir(main_folder_path):
#         dataset_path = os.path.join(main_folder_path, dataset_name)
        
#         if os.path.isdir(dataset_path):  # 如果是文件夹
#             print(f"\n处理数据集：{dataset_name}")
#             data = load_and_combine_csv_files(dataset_path)
            
#             if data is not None:
#                 analyze_correlation(data, dataset_name)
#                 analyze_feature_importance(data, dataset_name)

# # 加载并合并一个数据集文件夹内的所有CSV文件
# def load_and_combine_csv_files(dataset_path):
#     all_data = []
#     for file_name in os.listdir(dataset_path):
#         if file_name.endswith('.csv'):
#             file_path = os.path.join(dataset_path, file_name)
#             df = pd.read_csv(file_path)
#             all_data.append(df)
    
#     # 合并所有CSV文件
#     if all_data:
#         combined_data = pd.concat(all_data, ignore_index=True)
#         return combined_data
#     else:
#         print(f"没有找到CSV文件在：{dataset_path}")
#         return None

# # 相关性分析
# def analyze_correlation(data, dataset_name):
#     # 仅选择数值型列
#     numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
#     correlation_matrix = numeric_data.corr()  # 仅对数值列计算相关矩阵
    
#     # 绘制热力图
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
#     plt.title(f"Feature Correlation Matrix - {dataset_name}")
#     plt.savefig(f"{dataset_name}_correlation_matrix.png", dpi=300, format='png', bbox_inches='tight')
#     plt.close()  # 关闭图以释放内存
# # 特征重要性分析
# def analyze_feature_importance(data, dataset_name):
#     if 'Label' not in data.columns:
#         print(f"{dataset_name} 数据集中没有找到 'Label' 列，跳过特征重要性分析。")
#         return
    
#     # 仅保留数值型特征列
#     X = data.drop('Label', axis=1).select_dtypes(include=['float64', 'int64'])
#     y = data['Label']
    
#     # 检查是否仍有数值特征列
#     if X.empty:
#         print(f"{dataset_name} 数据集中没有数值型特征，跳过特征重要性分析。")
#         return
    
#     # 拆分训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
#     # 使用随机森林计算特征重要性
#     model = RandomForestClassifier()
#     model.fit(X_train, y_train)
#     feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
#     # 绘制特征重要性图
#     plt.figure(figsize=(10, 8))
#     feature_importances.plot(kind='bar')
#     plt.title(f"Feature Importance - {dataset_name}")
#     plt.savefig(f"{dataset_name}_feature_importance.png", dpi=300, format='png', bbox_inches='tight')  # 保存为高DPI图片
#     plt.close()  # 关闭图以释放内存


# # 运行主函数，替换 'path_to_main_folder' 为数据集主文件夹路径
# process_datasets('./csv_output/Tunnel')

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 主函数：处理所有数据集文件夹
def process_datasets(main_folder_path):
    # 遍历主文件夹下的所有子文件夹（每个代表一个数据集）
    for dataset_name in os.listdir(main_folder_path):
        dataset_path = os.path.join(main_folder_path, dataset_name)
        
        if os.path.isdir(dataset_path):  # 如果是文件夹
            print(f"\n处理数据集：{dataset_name}")
            data = load_and_combine_csv_files(dataset_path)
            
            if data is not None:
                analyze_correlation(data, dataset_name)
                analyze_feature_importance(data, dataset_name)

# 加载并合并一个数据集文件夹内的所有CSV文件
def load_and_combine_csv_files(dataset_path):
    all_data = []
    columns_to_drop = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort']
    
    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(dataset_path, file_name)
            df = pd.read_csv(file_path)
            
            # 移除不需要的列
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
            
            all_data.append(df)
    
    # 合并所有CSV文件
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    else:
        print(f"没有找到CSV文件在：{dataset_path}")
        return None

# 相关性分析
def analyze_correlation(data, dataset_name):
    # 仅选择数值型列
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    correlation_matrix = numeric_data.corr()  # 仅对数值列计算相关矩阵
    
    # 绘制热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
    plt.title(f"Feature Correlation Matrix - {dataset_name}")
    plt.savefig(f"{dataset_name}_correlation_matrix.png", dpi=300, format='png', bbox_inches='tight')
    plt.close()  # 关闭图以释放内存

# 特征重要性分析
def analyze_feature_importance(data, dataset_name):
    if 'Label' not in data.columns:
        print(f"{dataset_name} 数据集中没有找到 'Label' 列，跳过特征重要性分析。")
        return
    
    # 仅保留数值型特征列
    X = data.drop('Label', axis=1).select_dtypes(include=['float64', 'int64'])
    y = data['Label']
    
    # 检查是否仍有数值特征列
    if X.empty:
        print(f"{dataset_name} 数据集中没有数值型特征，跳过特征重要性分析。")
        return
    
    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 使用随机森林计算特征重要性
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    # 绘制特征重要性图
    plt.figure(figsize=(10, 8))
    feature_importances.plot(kind='bar')
    plt.title(f"Feature Importance - {dataset_name}")
    plt.savefig(f"{dataset_name}_feature_importance.png", dpi=300, format='png', bbox_inches='tight')  # 保存为高DPI图片
    plt.close()  # 关闭图以释放内存

# 运行主函数，替换 'path_to_main_folder' 为数据集主文件夹路径
process_datasets('./csv_output')
