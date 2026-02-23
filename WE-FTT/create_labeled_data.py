#!/usr/bin/env python3
"""
快速创建带标签的训练数据
根据文件名推断标签：f0=非地震(0), f1=地震(1)
"""

import pandas as pd
import sys
from pathlib import Path

def create_labeled_dataset(output_file='data/processed/training_data_labeled.parquet'):
    """创建带标签的训练数据"""

    print("正在加载数据文件...")

    # 加载数据并添加标签
    datasets = []

    # f0 文件 (label=0, 非地震)
    for i in range(5):  # t0-t4
        file_path = f'data/processed/downsampled_f0t{i}.csv'
        if Path(file_path).exists():
            print(f"加载 {file_path} (label=0)...")
            df = pd.read_csv(file_path)
            df['label'] = 0
            datasets.append(df)
            print(f"  加载了 {len(df)} 行")

    # f1 文件 (label=1, 地震)
    for i in range(5):  # t0-t4
        file_path = f'data/processed/downsampled_f1t{i}.csv'
        if Path(file_path).exists():
            print(f"加载 {file_path} (label=1)...")
            df = pd.read_csv(file_path)
            df['label'] = 1
            datasets.append(df)
            print(f"  加载了 {len(df)} 行")

    if not datasets:
        print("错误：未找到数据文件！")
        return None

    # 合并所有数据
    print("\n正在合并数据...")
    combined_df = pd.concat(datasets, ignore_index=True)
    print(f"合并后总共 {len(combined_df)} 行")

    # 打乱数据
    print("打乱数据...")
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 保存
    print(f"\n保存到 {output_file}...")
    combined_df.to_parquet(output_file, index=False)

    # 显示统计
    print("\n数据集统计:")
    print(f"总样本数: {len(combined_df)}")
    print(f"特征列: {len([c for c in combined_df.columns if 'BT_' in c and 'weight' not in c])}")
    print(f"权重列: {len([c for c in combined_df.columns if 'weight' in c])}")
    print(f"\n标签分布:")
    print(combined_df['label'].value_counts())
    print(f"\n列名: {list(combined_df.columns)}")

    return combined_df

if __name__ == '__main__':
    df = create_labeled_dataset()
    if df is not None:
        print("\n✓ 带标签的训练数据创建成功！")
    else:
        sys.exit(1)
