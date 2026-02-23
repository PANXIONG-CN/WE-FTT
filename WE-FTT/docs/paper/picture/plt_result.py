import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


data = {
    "Model": ["Random Forest", "LightGBM", "Catboost", "Xgboost", "TabNet", "Ours(WE-FTT)"]*6,
    "Metric": ["Accuracy"]*6 + ["Precision"]*6 + ["Recall"]*6 + ["F1"]*6 + ["Cohen's Kappa"]*6 + ["MCC"]*6,
    "Score": [
        0.7188, 0.5311, 0.5809, 0.4733, 0.6303, 0.9019,
        0.7109, 0.5628, 0.6176, 0.4806, 0.6405, 0.8048,
        0.7188, 0.5311, 0.5809, 0.4733, 0.6303, 0.9019,
        0.7001, 0.5001, 0.5809, 0.4925, 0.6229, 0.8163,
        0.6824, 0.5914, 0.5253, 0.5011, 0.5818, 0.8210,
        0.7404, 0.6044, 0.5328, 0.5151, 0.5917, 0.8367
    ]
}

df = pd.DataFrame(data)

palette = sns.color_palette("husl", n_colors=6)
mcc_color = "#d62728"

plt.figure(figsize=(18, 8))
sns.set_style("whitegrid")
ax = plt.gca()

models = df["Model"].unique()
metrics = df["Metric"].unique()
n_models = len(models)
n_metrics = len(metrics)

bar_width = 0.12
group_width = bar_width * n_metrics
gap_width = 0.15
total_width = group_width + gap_width

group_positions = np.arange(n_models) * total_width


for i, metric in enumerate(metrics):
    subset = df[df["Metric"] == metric]

    if metric == "MCC":
        color = mcc_color
        edgecolor = 'darkred'
        linewidth = 2
        alpha = 1.0
        label_fontweight = 'bold'
    else:
        color = palette[i]
        edgecolor = 'white'
        linewidth = 1
        alpha = 0.8
        label_fontweight = 'normal'

    positions = group_positions + i * bar_width
    bars = ax.bar(positions, subset["Score"], width=bar_width,
                  label=metric if i == 0 else "",
                  color=color, alpha=alpha,
                  edgecolor=edgecolor, linewidth=linewidth)

    for bar, model in zip(bars, models):
        height = bar.get_height()

        if height > 0.85:
            va = 'top'
            y_pos = height - 0.02
            color = 'white' if metric == "MCC" else 'black'
        else:
            va = 'bottom'
            y_pos = height + 0.01
            color = 'black'

        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{height:.3f}', ha='center', va=va,
                fontsize=10, color=color, weight=label_fontweight,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))

handles = []
labels = []
for i, metric in enumerate(metrics):
    if metric == "MCC":
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=mcc_color,
                       alpha=1.0, linewidth=2, edgecolor='darkred'))
    else:
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=palette[i], alpha=0.8))
    labels.append(metric)


ax.set_xticks(group_positions + group_width/2 - bar_width/2)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel('Score', fontsize=14)
ax.set_ylim(0, 1.05)
ax.set_title('Model Performance Comparison (MCC Highlighted)',
             fontsize=16, pad=20)


legend = ax.legend(handles, labels, title='Metrics',
                   bbox_to_anchor=(1.02, 1), loc='upper left',
                   frameon=True, fontsize=11)
legend.get_title().set_fontsize(12)

for text in legend.get_texts():
    if text.get_text() == "MCC":
        text.set_weight("bold")

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
ax.axhline(y=0.8, color='green', linestyle=':', linewidth=0.7, alpha=0.6)

plt.xlim(-0.3, n_models * total_width - gap_width + 0.3)

plt.tight_layout()
plt.savefig('/home/panxiong/MBT/Other/picture/model_performance_comparison.png',
            dpi=300, bbox_inches='tight')
plt.show()
