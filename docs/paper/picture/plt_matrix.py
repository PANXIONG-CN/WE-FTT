import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

output_dir = "/home/panxiong/MBT/Other/picture"
os.makedirs(output_dir, exist_ok=True)

confusion_matrix = np.array([
    [20363475, 126826, 693345, 11796, 118861,
        54921, 29437, 145092, 221021, 169275],
    [121122, 8176226, 24688, 188055, 2158, 152359, 169894, 122685, 1094, 2197],
    [57563, 1005, 5567345, 702, 5605, 91113, 98910, 30918, 459, 8057],
    [621603, 5235, 7711, 9866617, 14361, 16419, 355414, 471, 202627, 28081],
    [21220, 32398, 4186, 437, 9598887, 1306475, 3798, 56664, 149832, 2213],
    [630, 6444, 925, 702790, 35012, 20363380, 654551, 11423, 154968, 3813],
    [72, 15, 925, 209717, 3416, 123, 9082987, 165, 141106, 429983],
    [22, 2, 157288, 2, 11799, 49, 30439, 5068953, 95029, 4],
    [3, 152, 17001, 85421, 878497, 353, 205298, 355, 10067103, 64933],
    [27182, 3150, 492636, 74107, 10367, 64900, 27231, 730661, 146982, 11098846]
])

classes = [f"Class {i}" for i in range(10)]

# 1. 修改后的归一化混淆矩阵（不取log）
plt.figure(figsize=(15, 12))
norm_cm = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
sns.heatmap(norm_cm, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Normalized Value'})
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.savefig(os.path.join(
    output_dir, "1_normalized_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()

threshold = confusion_matrix.sum() * 0.005
error_flows = []
for i in range(10):
    for j in range(10):
        if i != j and confusion_matrix[i, j] > threshold:
            error_flows.append({
                'From': f'Class {i}',
                'To': f'Class {j}',
                'Count': confusion_matrix[i, j],
                'Percentage': confusion_matrix[i, j]/confusion_matrix.sum()*100
            })

df_errors = pd.DataFrame(error_flows).sort_values(
    'Count', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(data=df_errors, x='Percentage', y='From', hue='To',
            palette='viridis', dodge=False)
plt.title("Top 20 Misclassification Flows (>0.5% of Total Samples)")
plt.xlabel("Percentage of Total Samples (%)")
plt.ylabel("Source Class")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "2_top_error_flows.png"),
            dpi=300, bbox_inches='tight')
plt.close()

metrics = ["Precision", "Recall", "Diagonal Accuracy"]
class_data = []
for i in range(10):
    tp = confusion_matrix[i, i]
    fp = confusion_matrix[:, i].sum() - tp
    fn = confusion_matrix[i, :].sum() - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    diag_acc = tp / confusion_matrix[i, :].sum()
    class_data.append([precision, recall, diag_acc])

angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)

for i in range(10):
    values = class_data[i] + [class_data[i][0]]
    ax.plot(angles, values, linewidth=2, linestyle='solid',
            label=f'Class {i}', alpha=0.7)
    ax.fill(angles, values, alpha=0.1)

plt.xticks(angles[:-1], metrics)
plt.title("Class-wise Performance Metrics Radar Chart", pad=20)
plt.legend(bbox_to_anchor=(1.1, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "3_class_radar.png"),
            dpi=300, bbox_inches='tight')
plt.close()

# 4. 跨频段混淆分析
freq_bands = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
cross_band = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        if not any((i in band) and (j in band) for band in freq_bands):
            cross_band[i, j] = confusion_matrix[i, j]

plt.figure(figsize=(12, 10))
sns.heatmap(cross_band / confusion_matrix.sum(axis=1, keepdims=True),
            annot=True, fmt=".1%", cmap="Blues",
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Confusion Percentage'})
plt.title(
    "Cross-Frequency Band Confusion Rates\n(Grouping: [0,1], [2,3], [4,5], [6,7], [8,9])")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "4_cross_band_confusion.png"),
            dpi=300, bbox_inches='tight')
plt.close()

metrics_data = []
for i in range(10):
    tp = confusion_matrix[i, i]
    fp = confusion_matrix[:, i].sum() - tp
    fn = confusion_matrix[i, :].sum() - tp
    metrics_data.append({
        "Class": f"Class {i}",
        "Precision": f"{tp/(tp+fp):.1%}",
        "Recall": f"{tp/(tp+fn):.1%}",
        "F1-Score": f"{2*tp/(2*tp+fp+fn):.1%}"
    })

df = pd.DataFrame(metrics_data)

plt.figure(figsize=(10, 4))
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
pd.plotting.table(ax, df, loc='center', cellLoc='center')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "5_metrics_table.png"),
            dpi=300, bbox_inches='tight')
plt.close()

print(f"所有图表已保存至: {output_dir}")
