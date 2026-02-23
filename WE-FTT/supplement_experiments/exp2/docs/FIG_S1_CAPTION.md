# Figure S1 Caption
# Figure S1 题注：时间演化和性能时间线

---

## Figure S1 | Temporal evolution and performance timeline of out-of-sample events

### English

Temporal evolution of model performance on forward-looking validation demonstrates robust temporal stability across post-training earthquake events from USGS catalog. The figure comprises two panels: **(a) Temporal performance timeline** displays the distribution of 26 real M≥7.0 earthquakes occurring between August 2023 and September 2025, with vertical lines representing event timing and circle sizes proportional to earthquake magnitude (M7.0–8.8). Color coding indicates Matthews Correlation Coefficient (MCC) performance, ranging from 0.75 (red, representing lower performance) to 0.82 (green, representing higher performance). Individual MCC values are annotated above selected events for precise quantification. The colorbar on the right provides the MCC scale reference. **(b) Zone-averaged performance** presents bar charts showing mean MCC values across different environmental zones (A-E), with error bars indicating standard deviations. Bar colors correspond to zone-specific color schemes defined in the main text. The horizontal dashed red line marks the in-sample baseline performance (MCC = 0.840), while the solid blue line indicates the out-of-sample mean (MCC = 0.802).

**Statistical Analysis**: The analysis reveals good temporal stability, with overall out-of-sample performance showing 4.5% degradation from baseline (MCC: 0.840 → 0.802, Δ = -0.038). The consistency of performance metrics across the two-year span confirms model stability: F1 score mean = 0.788 (SD = 0.009), Precision mean = 0.767 (SD = 0.008), Recall mean = 0.808 (SD = 0.008). All environmental zones maintain MCC values above 0.77, validating the environment-specific approach across diverse tectonic settings including subduction zones (Kamchatka, Alaska), collision boundaries (Myanmar), and oceanic transform faults (Drake Passage). The preservation of high recall (mean = 0.808) demonstrates appropriate prioritization of sensitivity for operational earthquake early warning applications, while the slight decrease in precision (mean = 0.767) reflects the inherent trade-off in out-of-sample generalization.

**Key Findings**: (1) Minimal temporal drift with only 4.5% performance degradation over 2 years; (2) Consistent performance across 26 diverse earthquake events; (3) Stable performance across different environmental zones; (4) Recall preservation indicates maintained sensitivity for early warning; (5) No evidence of overfitting, as degradation is modest and consistent.

---

### 中文

模型在前瞻性验证上的性能时间演化展示了跨USGS目录训练后地震事件的稳健时间稳定性。图表包含两个面板：**(a) 时间性能时间线**显示了2023年8月至2025年9月期间发生的26个真实M≥7.0地震的分布，垂直线表示事件时间，圆圈大小与地震震级（M7.0–8.8）成比例。颜色编码表示Matthews相关系数（MCC）性能，范围从0.75（红色，代表较低性能）到0.82（绿色，代表较高性能）。选定事件上方标注了单独的MCC值以进行精确量化。右侧颜色条提供MCC刻度参考。**(b) 区域平均性能**显示了不同环境区域（A-E）的平均MCC值条形图，误差条表示标准偏差。条形颜色对应主文中定义的区域特定配色方案。水平虚线红线标记样本内基线性能（MCC = 0.840），而实线蓝线表示样本外平均值（MCC = 0.802）。

**统计分析**：分析揭示了良好的时间稳定性，总体样本外性能显示出较基线4.5%的退化（MCC：0.840 → 0.802，Δ = -0.038）。性能指标在两年跨度上的一致性确认了模型稳定性：F1分数平均值 = 0.788（SD = 0.009），精度平均值 = 0.767（SD = 0.008），召回率平均值 = 0.808（SD = 0.008）。所有环境区域都保持MCC值在0.77以上，验证了环境特定方法在包括俯冲带（堪察加、阿拉斯加）、碰撞边界（缅甸）和海洋转换断层（德雷克海峡）在内的不同构造环境中的有效性。高召回率的保持（平均值 = 0.808）展示了对操作性地震早期预警应用灵敏度的适当优先级，而精度的轻微下降（平均值 = 0.767）反映了样本外泛化中固有的权衡。

**关键发现**：（1）最小的时间漂移，2年内仅4.5%的性能退化；（2）跨26个不同地震事件的一致性能；（3）跨不同环境区域的稳定性能；（4）召回率保持表明维持了早期预警的灵敏度；（5）无过拟合迹象，因为退化幅度适度且一致。

---

## Technical Specifications

- **Figure Format**: PDF (vector), PNG (raster, 600 DPI), SVG (editable vector)
- **Figure Size**: Double-column width (7.2 inches / 183 mm)
- **Font**: Serif family (DejaVu Serif, Times New Roman, etc.)
- **Math Font**: STIX
- **Color Scheme**:
  - MCC gradient: Red (#E15759) → Yellow (#EDC948) → Green (#59A14F)
  - Zone colors: As defined in main text Figure 2
- **Sub-panel Labels**: (a) and (b) in bold, 12pt font

---

**Figure Generation**: Generated using Python 3.9+ with matplotlib 3.x and Nature journal style guidelines. Source code: `exp2/scripts/generate_fig_s1.py`.

**Data Source**: USGS Earthquake Catalog (https://earthquake.usgs.gov/), queried for M≥7.0 events from 2023-08-01 to 2025-09-30.
