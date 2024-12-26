import os
import pandas as pd
from collections import defaultdict
import json
from tqdm import tqdm

# 定义根目录
root_dir = 'csv_output'

# 用于存储每个目录下的Label分布
label_distribution_per_directory = defaultdict(lambda: defaultdict(int))

# 使用tqdm显示处理进度
for current_dir, sub_dirs, files in tqdm(os.walk(root_dir), desc="Processing directories"):
    # 遍历目录下的每个文件
    for file in tqdm(files, desc=f"Processing files in {current_dir}", leave=False):
        if file.endswith('.csv'):  # 只处理CSV文件
            file_path = os.path.join(current_dir, file)
            try:
                # 检查文件是否为空
                if os.path.getsize(file_path) == 0:
                    print(f"Skipping empty file: {file_path}")
                    continue

                # 读取CSV文件
                df = pd.read_csv(file_path)

                # 检查是否存在数据和 'Label' 列
                if df.empty:
                    print(f"Skipping file with no data: {file_path}")
                    continue

                if 'Label' not in df.columns:
                    print(f"Skipping file with no 'Label' column: {file_path}")
                    continue

                # 统计Label列的值分布
                label_counts = df['Label'].value_counts()

                # 将统计结果记录到当前目录的计数中
                for label, count in label_counts.items():
                    label_distribution_per_directory[current_dir][label] += count

            except pd.errors.EmptyDataError:
                print(f"Error reading {file_path}: No columns to parse from file")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# 将结果保存为JSON文件
with open('label_distribution.json', 'w') as json_file:
    json.dump(label_distribution_per_directory, json_file, indent=4)

# 将结果保存为CSV文件
csv_output = []
for directory, label_distribution in label_distribution_per_directory.items():
    for label, count in label_distribution.items():
        csv_output.append([directory, label, count])

df_output = pd.DataFrame(csv_output, columns=['Directory', 'Label', 'Count'])
df_output.to_csv('label_distribution.csv', index=False)

print("Results have been saved to 'label_distribution.json' and 'label_distribution.csv'.")
