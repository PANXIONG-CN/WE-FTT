import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 确保输出目录存在
output_dir = "/home/panxiong/MBT/Other/picture"
os.makedirs(output_dir, exist_ok=True)

# 读取数据
data = []
with open("/home/panxiong/MBT/Other/Apriori_Results/normalized_MBTDATA_freqItemsets_type_4.txt", "r") as f:
    for line in f:
        data.append(json.loads(line))

# 转换为DataFrame并预处理
df = pd.DataFrame(data)
df['items'] = df['items'].apply(lambda x: sorted(x.split(',')))
df['num_items'] = df['items'].apply(len)
df['items_str'] = df['items'].apply(lambda x: ", ".join(x))

# 筛选Top 20高频项集
top_df = df.sort_values('support_diff', ascending=False).head(20).copy()

# --------------------- 1. 热力图 ---------------------
# 提取所有唯一项
all_items = list(set(item for sublist in df['items'] for item in sublist))

# 创建热力图矩阵
heatmap_data = pd.DataFrame(0, index=all_items, columns=all_items)
for _, row in df.iterrows():
    items = row['items']
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            heatmap_data.loc[items[i], items[j]] += row['support_diff']
            heatmap_data.loc[items[j], items[i]] += row['support_diff']

# 筛选高频交互项（至少出现3次）
item_counts = (heatmap_data > 0).sum(axis=1)
selected_items = item_counts[item_counts >= 3].index.tolist()
heatmap_data_filtered = heatmap_data.loc[selected_items, selected_items]

# 自定义颜色
cmap = LinearSegmentedColormap.from_list(
    'custom', ['#f7fbff', '#6baed6', '#08519c', '#08306b'], N=256)

plt.figure(figsize=(16, 14))
sns.heatmap(heatmap_data_filtered,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            linewidths=0.5,
            cbar_kws={'label': 'Cumulative Support Difference'},
            annot_kws={'size': 8})
plt.title("Zone E: Association Strength Between Frequent Items (Top Interactions)",
          pad=20, fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(f"{output_dir}/heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# --------------------- 2. 条形图 ---------------------
plt.figure(figsize=(12, 8))
barplot = sns.barplot(x='support_diff',
                      y='items_str',
                      data=top_df,
                      palette="viridis",
                      edgecolor='black',
                      linewidth=0.5)

# 添加数值标签
for i, (_, row) in enumerate(top_df.iterrows()):
    barplot.text(row['support_diff'] + 0.01, i,
                 f"{row['support_diff']:.3f}",
                 va='center',
                 fontsize=10)

plt.xlabel("Support Difference", fontsize=12)
plt.ylabel("Frequent Itemset", fontsize=12)
plt.title("Zone E: Top 20 Frequent Itemsets by Support Difference", fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"{output_dir}/barplot.png", dpi=300, bbox_inches='tight')
plt.close()


print(f"可视化结果已保存至: {output_dir}/")
print("生成文件: heatmap.png, barplot.png, network.png")
