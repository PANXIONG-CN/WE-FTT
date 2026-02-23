import matplotlib.pyplot as plt
import re
import os
import numpy as np
from matplotlib.ticker import MaxNLocator

# 设置输出目录
output_dir = "/home/panxiong/MBT/Other/picture"
os.makedirs(output_dir, exist_ok=True)

# 读取日志文件
with open("/home/panxiong/MBT/Other/ablation_nohup.txt", "r") as f:
    log_data = f.read()

# 解析实验组
experiments = re.split(
    r'=== Starting (.*?) ablation: (.*?) ===\n', log_data)[1:]
experiment_groups = []
for i in range(0, len(experiments), 3):
    group_name = f"{experiments[i]}\n({experiments[i+1]})"  # 添加换行符使标题更清晰
    logs = experiments[i+2].strip().split('\n')
    experiment_groups.append((group_name, logs))

# 创建大图 (调整为5行3列布局，共15个子图位置，实际使用13个)
plt.figure(figsize=(24, 30))
plt.suptitle("Ablation Study Results (13 Experiments)",
             y=0.995, fontsize=18, fontweight='bold')

# 使用tab20色系 (确保有足够颜色)
colors = plt.cm.tab20(np.linspace(0, 1, 20))

for idx, (group_name, logs) in enumerate(experiment_groups, 1):
    # 解析日志数据
    epochs, acc, f1, mcc = [], [], [], []
    for log in logs:
        parts = re.match(
            r'Epoch\s+(\d+)/\d+ \| Loss: [\d.]+ \| Accuracy: ([\d.]+) \| F1: ([\d.]+) \| MCC: ([\d.]+)',
            log
        )
        if parts:
            epochs.append(int(parts.group(1)))
            acc.append(float(parts.group(2)))
            f1.append(float(parts.group(3)))
            mcc.append(float(parts.group(4)))

    # 创建子图 (5行3列布局)
    ax = plt.subplot(5, 3, idx)

    # 安全获取颜色 (防止索引越界)
    color_idx = (idx-1) % len(colors)

    # 绘制三条曲线
    line_acc, = ax.plot(epochs, acc, 'o-', color=colors[color_idx],
                        label='Accuracy', linewidth=2, markersize=6)
    line_f1, = ax.plot(epochs, f1, 's--', color=colors[(color_idx+1) % len(colors)],
                       label='F1 Score', linewidth=2, markersize=6, alpha=0.8)
    line_mcc, = ax.plot(epochs, mcc, '^-.', color=colors[(color_idx+2) % len(colors)],
                        label='MCC', linewidth=2, markersize=6, alpha=0.6)

    # 设置子图标题和标签
    ax.set_title(group_name, fontsize=12, pad=10)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 自动调整y轴范围
    min_val = min(min(acc), min(f1), min(mcc))
    max_val = max(max(acc), max(f1), max(mcc))
    ax.set_ylim([max(0, min_val - 0.05), min(1, max_val + 0.05)])

    # 仅在第一个子图显示图例
    if idx == 1:
        ax.legend(handles=[line_acc, line_f1, line_mcc],
                  fontsize=10, loc='upper left',
                  bbox_to_anchor=(1.05, 1))

# 调整布局 (增加子图间距)
plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=4.0)

# 保存图像
output_path = os.path.join(output_dir, "ablation_study_comparison.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"消融实验对比图已保存至: {output_path}")
