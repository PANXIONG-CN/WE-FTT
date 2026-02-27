# 修订任务 03：主文与补充材料改写（revisions-3nd）

> 目标：在 `WE-FTT/docs/revision_codex/revisions-3nd/` 内完成第三轮改稿的工作区初始化与后续落稿清单，严格按 `WE-FTT/docs/revision_codex/ai-reviews/comprehensive-analysis-and-execution-plan.md`（T9/T11）收缩主张、重构叙事。

## 0. 本任务的工作区（已初始化）

已创建并拷贝“最小可编译集”：
- 目录：`WE-FTT/docs/revision_codex/revisions-3nd/`
- 主文基线：`WE-FTT/docs/revision_codex/revisions-3nd/main_clean.tex`（来源 `revisions-2nd/main_rev.tex`）
- 参考文献：`WE-FTT/docs/revision_codex/revisions-3nd/references.bib`
- 模板文件：`WE-FTT/docs/revision_codex/revisions-3nd/elsarticle.cls`、`WE-FTT/docs/revision_codex/revisions-3nd/elsarticle-harv.bst`
- 图件：同步拷贝了 `main_clean.tex` 当前引用的全部 `image*.png`、`fig_*.pdf/png`

> 约束：本轮不拷贝/不依赖 `build/`；所有修改仅在 `revisions-3nd/` 内进行，保持 `revisions-2nd/` 可追溯不变。

## 1. 改写范围（已决策）

- 执行 **T11 全面叙事重构**（不是仅替换措辞）
- 但不强制进行“主文/补充材料”大规模重编号；新增内容从 S5+ 追加

## 2. 关键插入点（对齐补强实验）

> 实验结果由任务 01 产出；表图接入由任务 02 负责；本任务负责把它们写进正文与补充材料。

必须在正文中明确呈现的“审稿人会抓住的硬证据”：
- [x] T1：权重/聚类/关联规则无泄漏（给出可审计说明 + 评估协议链接）
- [x] T2：事件级切分（防止同事件跨集合）
- [x] T2-补充：空间外推压力测试按“边界结果”叙事落稿（不主张显著提升；强调跨区迁移有限但 FPR 可控）
- [x] T4：Placebo（零分布 + p 值，p<0.01）
- [x] T6：ERA5 海洋条件化后 FPR 显著下降（目标 <0.10）
- [x] T5：噪声预算 + 时空聚合降噪（支撑 Table S5 的“方法学动机翻转”）

## 3. 措辞替换清单（必须执行，按 T11）

建议以“搜索 → 逐处语义改写”而非机械替换：

```
"reliable detection" → "statistically significant association"
"near-perfect" → 删除或改为 "strong" / "consistent"
"operational early warning" → "monitoring protocol" / 删除
"precursor" → 多数改为 "earthquake-window-associated anomaly"
"deep crustal stress" → 删除或改为 "near-surface coupling"（且加 caveat）
"strongly support" → "is consistent with" / "suggests"
"quasi-tomographic" → 删除
```

## 4. 标题与贡献收缩（必须对齐 TGRS）

- [x] Title/Abstract：从“precursor detection”收缩为“anomaly characterization/evaluation protocol”
- [x] Introduction：问题定义从“单像元可测”转为“强混杂背景下的可检验评估”
- [x] Results：前瞻验证（26 事件）提升为主证据叙事（不必搬动图号）
- [x] Discussion：机制描述一律作为 *possible surface-coupling pathway*，并强调浅表敏感性限制
- [x] Limitations：明确单传感器局限、海洋高 FPR、适用边界（M≥7.0、特定环境）

## 5. Supplementary（同一 tex 内）新增条目与编号

在 `\section*{Supplementary Figures and Tables}` 的 itemize 中追加：
- [x] Figure S5：Placebo MCC 零分布
- [x] Figure S6：ERA5 海洋条件化前后对比
- [x] Table S5：Fresnel/介电灵敏度对比（重新包装为方法学动机）
- [x] Table/ Figure S7+：按任务 01/02 实际产物补齐

并确保：
- [x] 计数器 reset 逻辑保持正确（已存在 `\setcounter{figure}{0}` 等）
- [x] `\label{fig:figS5}` / `\label{tab:tableS5}` 等与引用一致

## 6. 验收标准（DoD）

### 6.1 结构与可编译
- [x] `revisions-3nd/main_clean.tex` 可独立编译（不依赖 `revisions-2nd/build/`）
- [x] 全部 `\includegraphics{...}` 文件存在于 `revisions-3nd/`

### 6.2 叙事与风险控制
- [x] 全文（含题注）不再出现：`near-perfect` / `operational early warning` 等禁用词
- [x] Placebo 与 ERA5 条件化结果在正文有“可检验的主张”（带 p/CI/样本量）
- [x] Table S5 从“不可测自证”翻转为“方法学动机 + 聚合降噪论证”

### 6.3 可复现承诺一致性
- [x] 文稿中对“评估协议公开”的描述与 `WE-FTT/evaluation_protocol/` 的实际结构一致
