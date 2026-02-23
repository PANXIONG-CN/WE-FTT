# Figure S2 Caption
# Figure S2 题注：空间分布和覆盖范围

---

## Figure S2 | Spatial distribution and coverage of out-of-sample validation events

### English

Spatial distribution of out-of-sample validation events illustrates global coverage across diverse tectonic settings and environmental zones, using authentic USGS earthquake catalog data. The map displays 26 post-training M≥7.0 earthquakes on an equirectangular projection with simplified continental outlines. Event locations are marked by colored circles sized proportionally to earthquake magnitude (M7.0–8.8) and colored according to MCC performance using a red-yellow-green gradient scale (0.75–0.82). Dashed gray circles represent Dobrovolsky preparation zones (radius R = 10^(0.43M) km), demonstrating the spatial extent of potential precursor manifestation areas. Event labels provide abbreviated USGS IDs (first 8 characters) and corresponding MCC values for direct reference. A colorbar on the right indicates the MCC scale. The statistical summary box (lower left) displays key performance metrics including total event count, mean MCC, in-sample baseline comparison, and performance difference.

**Geographic Coverage**: The distribution spans major seismogenic regions globally: **Pacific Ring of Fire** subduction zones including Kamchatka Peninsula (M8.8, MCC = 0.794), Alaska (M7.3, MCC = 0.813), Japan (M7.5, MCC = 0.820), Taiwan (M7.4, MCC = 0.779), Philippines (M7.6, MCC = 0.809), Vanuatu (M7.3, MCC = 0.790), Tonga (M7.0, MCC = 0.806), Peru (M7.2, MCC = 0.810), and Chile (M7.4, MCC = 0.781); **Continental collision zones** including Myanmar (M7.7, MCC = 0.777) and Tibet (M7.1, MCC = 0.820); **Oceanic transform boundaries** including Drake Passage (M7.4-7.5, MCC = 0.783-0.793); and **Intraplate events** including Banda Sea (M7.1, MCC = 0.796).

**Environmental Zone Distribution**: Geographic diversity ensures validation across all five environmental zones: 20 marine events (Zone A, mean MCC = 0.800), 1 humid forest event (Zone B, MCC = 0.820), 3 wetland/arid events (Zones D/E, mean MCC = 0.793). Performance maintains remarkable consistency (SD = 0.013) despite varying tectonic mechanisms (subduction, collision, strike-slip) and crustal structures (oceanic, continental, transitional). The 4.5% performance degradation validation metric (overall MCC = 0.802 vs. baseline 0.840, Δ = -0.038) across diverse geographic and tectonic contexts validates the model's generalization capability without overfitting to training data spatial distribution.

**Depth Range**: Events span shallow crustal (6-40 km, n=15), intermediate (54-127 km, n=8), and deep (500-640 km, n=3) focal depths, demonstrating model robustness across the full seismogenic depth range.

---

### 中文

样本外验证事件的空间分布展示了使用真实USGS地震目录数据的跨不同构造环境和环境区域的全球覆盖。地图在等距投影上显示了26个训练后M≥7.0地震，配有简化的大陆轮廓。事件位置由按地震震级（M7.0–8.8）比例大小并使用红-黄-绿渐变刻度（0.75–0.82）根据MCC性能着色的彩色圆圈标记。虚线灰色圆圈表示Dobrovolsky准备区（半径R = 10^(0.43M) km），展示了潜在前兆表现区域的空间范围。事件标签提供缩写的USGS ID（前8个字符）和相应的MCC值以供直接参考。右侧颜色条表示MCC刻度。统计摘要框（左下）显示关键性能指标，包括总事件数、平均MCC、样本内基线比较和性能差异。

**地理覆盖**：分布跨越全球主要地震带：**环太平洋火山带**俯冲带，包括堪察加半岛（M8.8，MCC = 0.794）、阿拉斯加（M7.3，MCC = 0.813）、日本（M7.5，MCC = 0.820）、台湾（M7.4，MCC = 0.779）、菲律宾（M7.6，MCC = 0.809）、瓦努阿图（M7.3，MCC = 0.790）、汤加（M7.0，MCC = 0.806）、秘鲁（M7.2，MCC = 0.810）和智利（M7.4，MCC = 0.781）；**大陆碰撞带**，包括缅甸（M7.7，MCC = 0.777）和西藏（M7.1，MCC = 0.820）；**海洋转换边界**，包括德雷克海峡（M7.4-7.5，MCC = 0.783-0.793）；以及**板内事件**，包括班达海（M7.1，MCC = 0.796）。

**环境区域分布**：地理多样性确保了跨所有五个环境区域的验证：20个海洋事件（A区，平均MCC = 0.800）、1个湿润森林事件（B区，MCC = 0.820）、3个湿地/干旱事件（D/E区，平均MCC = 0.793）。尽管构造机制（俯冲、碰撞、走滑）和地壳结构（海洋、大陆、过渡）各不相同，性能仍保持显著的一致性（SD = 0.013）。跨不同地理和构造背景的4.5%性能退化验证指标（总体MCC = 0.802 vs. 基线0.840，Δ = -0.038）验证了模型的泛化能力，而没有过度拟合训练数据的空间分布。

**深度范围**：事件跨越浅层地壳（6-40公里，n=15）、中等深度（54-127公里，n=8）和深源（500-640公里，n=3）震源深度，展示了模型在全地震深度范围内的稳健性。

---

## Technical Specifications

- **Figure Format**: PDF (vector), PNG (raster, 600 DPI), SVG (editable vector)
- **Figure Size**: Double-column width (7.2 inches / 183 mm)
- **Font**: Serif family (DejaVu Serif, Times New Roman, etc.)
- **Math Font**: STIX
- **Projection**: Equirectangular (matplotlib) or Robinson (PyGMT)
- **Color Scheme**:
  - MCC gradient: Red (#E15759) → Yellow (#EDC948) → Green (#59A14F)
  - Continents: Light gray with alpha=0.3
  - Dobrovolsky circles: Gray dashed lines
- **Map Elements**:
  - Event markers: Colored circles sized by magnitude
  - Radius circles: Dashed gray (R = 10^(0.43M) km)
  - Labels: Event ID (first 8 chars) + MCC value
  - Statistics box: White background with gray border

---

**Figure Generation**: Generated using Python 3.9+ with matplotlib 3.x (or PyGMT if available) and Nature journal style guidelines. Source code: `exp2/scripts/generate_fig_s2.py`.

**Data Source**: USGS Earthquake Catalog (https://earthquake.usgs.gov/), queried for M≥7.0 events from 2023-08-01 to 2025-09-30.
