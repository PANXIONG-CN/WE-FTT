# 审稿意见回复规范（Response to Reviewers Guideline）

> 适用对象：准备向期刊提交返修稿的作者团队  
> 默认场景：LaTeX 模板 `response_to_reviewers.tex`，以 Remote Sensing of Environment (RSE) 为示例，但可推广到其他期刊

---

## 1. 目标与适用范围

- 本规范用于指导撰写**审稿意见回复文件**（Response to Reviewers），确保：
  - 对每一条审稿意见都有**完整、清晰、可追踪**的回复；
  - 回复与**论文实际修改**一一对应，方便 AE/Reviewers 检查；
  - 风格专业、礼貌，符合国际期刊常规。
- 默认技术栈：
  - 主文件：`response_to_reviewers.tex`（LaTeX）；
  - 与主稿件、补充材料一起提交；
  - 可根据期刊要求导出为 PDF 上传。

---

## 2. 文件结构与命名约定

- 建议文件命名：
  - 主稿：`YYYYMMDD-main-elsarticle.tex`（或期刊模板）；
  - 回复文件：`response_to_reviewers.tex`；
  - 对应 PDF：`response_to_reviewers.pdf`。
- `response_to_reviewers.tex` 推荐结构：

  1. Cover Letter（致 Editor/AE 的简短总述信）
  2. `\subsection*{Response to Associate Editor}`
  3. `\subsection*{Response to Reviewer #1}`
  4. `\subsection*{Response to Reviewer #2}`
  5. … 依次扩展更多审稿人

- 在每个 AE/Reviewer 区块内部，对每条评论使用**统一三段式结构**：
  1. 原始意见（Comment）——一字不删；
  2. 综合回复（Response）——概括性、结构化回答；
  3. 修改说明（Change in the manuscript）——指出在稿件中**具体改了什么**。

---

## 3. LaTeX 模板、颜色与引用格式约定

### 3.1 颜色与宏定义（必须保持一致）

在 `response_to_reviewers.tex` 的导言区推荐定义：

```tex
\definecolor{commentcolor}{rgb}{0.0, 0.0, 0.5}
\definecolor{responsecolor}{rgb}{0.0, 0.5, 0.0}
\definecolor{changecolor}{rgb}{0.5, 0.0, 0.0}

\newcommand{\revcomment}[1]{\begingroup\color{commentcolor}#1\endgroup}
\newcommand{\revresponse}[1]{\begingroup\color{responsecolor}#1\endgroup}
\newcommand{\revchange}[1]{\begingroup\color{changecolor}#1\endgroup}
```

规范要求：

- `\revcomment{...}`：
  - 仅用于包裹**审稿人/AE 原始评论**；
  - 使用 `commentcolor`（深蓝色），视觉上与作者内容区分；
  - **不得修改评论内容**（包括词汇、语序、语法错误等）。
- `\revresponse{...}`：
  - 用于作者的**综合回复**；
  - 使用 `responsecolor`（绿色），便于 reviewers 快速识别。
- `\revchange{...}`：
  - 用于描述**在稿件中实际做了哪些修改**；
  - 使用 `changecolor`（深红色），强调“改动”内容；
  - 对每一条实质修改，必须写在 `\revchange{...}` 中。

除上述三个宏外：

- 不再额外引入其他颜色宏，以免样式混乱；
- 不直接在正文中使用 `\textcolor{...}{...}` 替代上述宏。

### 3.2 引用格式（页码、章节、图表）

在 `\revchange{...}` 中引用稿件内容时，统一格式：

- 章节/子节：
  - `Section~2.3`, `Subsection~3.6`；
- 页码：
  - `p.~15`（英文期刊中常用形式）；
  - 示例：`In Section~3.6 (p.~28), we added a new paragraph that ...`
- 图表：
  - 正文图：`Figure~3`, `Figures~8--10`；
  - 正文表：`Table~2`；
  - 补充材料：`Figure~S1`, `Table~S3`。
- 引文：
  - 文内引用使用期刊指定格式（例如 natbib）：
    - `\citet{Shah2015}` 或 `\citep{Shah2015}`；
  - 在 `\revchange` 中如果需要强调新增/修改的参考文献：
    - 可简要列出：`We added Shah et al.~(2015, 2020, 2022) to the reference list and cited them in Sections~3.6 and 4.2.2.`

---

## 4. Cover Letter 写作规范（可复用模板）

Cover Letter 是 `response_to_reviewers.tex` 的开头部分，用于向 Editor/AE 总结修回工作的关键内容。

### 4.1 结构要求

推荐结构：

1. 顶部日期 + 收件人信息（Editor-in-Chief / Journal name 等）
2. 问候语 + 稿件基本信息（题目、类型）
3. 一段简短的总体感谢与修回说明
4. **高影响修改概览**（What we changed，通常 3–5 条 bullet）
5. 结果与定位（Outcome and positioning）
6. 行政声明（Administrative statements：原创性、冲突、数据与代码等）
7. 结尾礼貌用语 + 通讯作者签名与单位

### 4.2 英文模板示例

可作为基本骨架使用（按需替换期刊、题目、指标等）：

```tex
\begin{flushright}
\today
\end{flushright}

\bigskip

Editor-in-Chief\\
\textit{[Journal Name]}\\
[Publisher]

\bigskip

Dear Prof.~[Editor Name] and Associate Editor,

We are pleased to submit our revised manuscript entitled \textit{``[Manuscript Title]''}. We are grateful for the Associate Editor’s guidance and the reviewers’ thoughtful critiques. Following the ``[Decision Type]'' decision, we have undertaken substantial revisions that we believe address all concerns with concrete analyses, transparent statistics, and precise manuscript edits.

\medskip

\noindent\textbf{What we changed (high-impact items):}

\begin{enumerate}
\item \textbf{[High-impact change 1].} Brief description of the major change, including what was added (e.g., new experiments, global validation, ablation studies) and where it is documented (e.g., new Section~3.6, Figure~13, Supplementary Figure~S1).
\item \textbf{[High-impact change 2].} ...
\item \textbf{[High-impact change 3].} ...
% (Optional more items)
\end{enumerate}

\medskip

\noindent\textbf{Outcome and positioning:}

[2--4 sentences summarizing overall performance, robustness, and how the revised manuscript positions the work in the literature or application domain.]

\medskip

\noindent\textbf{Administrative statements:}

\begin{itemize}
\item This submission is original and not under consideration elsewhere. All authors approve the content and order of authorship.
\item No conflicts of interest are declared.
\item Data/code: [brief statement on availability, e.g., will be shared upon request or via repository upon acceptance, in line with journal policy].
\item We upload: revised manuscript (clean), revised manuscript (marked-up), response-to-reviewers (this file), and Supplementary Information.
\end{itemize}

We appreciate the careful evaluation and hope these revisions meet the journal’s standards. We would be grateful if the revision could be handled by the same Associate Editor and reviewers, who are already familiar with the technical scope.

\bigskip

Sincerely,\\[0.5em]
\vspace{2em}

\textbf{[Corresponding Author Name]}\\
[Affiliation]\\
Email: \texttt{[email]}
```

---

## 5. 逐条意见回复的三段式规范（核心）

每一条 AE/Reviewer 意见（包括子问题）必须按照以下“三部分”回复：

1. **原始意见（Comment）**：完整复制审稿人文本，一字不删；
2. **综合回复（Response）**：作者的结构化回应，解释思路和结论；
3. **修改说明（Change in the manuscript）**：在稿件中具体改了什么，在哪里改。

### 5.1 LaTeX 三段式模板

推荐的基本结构（以 Reviewer #1, Comment R1-2 为例）：

```tex
\revcomment{%
\textbf{Comment R1-2.} [在此粘贴审稿人原话，一字不删。可以适当分段，但不要改动任何词汇、标点或语序。]
}

\revresponse{%
\textbf{Response (R1-2).} [在此写综合回复，遵循以下原则：
(i) 开头感谢并概括问题；
(ii) 明确说明是否同意（完全同意/部分同意/不同意并解释原因）；
(iii) 简要概述采取了哪些具体行动（新增实验、重写段落、澄清定义等）。]
}

\revchange{%
\textbf{Change (R1-2) in the manuscript.}
[在此详细列出修改内容，要求：
(1) 指出具体位置：章节名 + 页码 + 段落位置；
(2) 如是“实质修改”，必须贴出修改后的关键文本片段；
(3) 涉及图/表/补充材料的，必须给出编号和简要说明。]
}
```

注意：

- `\revcomment` 中只放 Reviewer 原文，不掺杂作者解释；
- `\revresponse` 中不再引用 Reviewer 原文，只用标号 `R1-2` 指代；
- `\revchange` 中所有修改都用**过去时**描述：`we added`, `we revised`, `we clarified`。

### 5.2 综合回复（Response）的写作要点

建议结构（英文）：

- 开头一句：
  - `We thank the reviewer for this insightful comment.`  
  - `We appreciate the reviewer’s careful assessment of [topic].`
- 接下来 1–3 句：
  - 概括 Reviewer 关切的核心点；
  - 说明作者是否同意，以及大致如何应对；
- 后续若干句：
  - 概述具体动作（将在 `\revchange` 中展开）；
  - 如有无法完全满足的要求，要说明原因（数据不可获得、超出篇幅、超出本研究范围等），并给出有限补救措施。

英文模板片段：

```tex
\revresponse{%
\textbf{Response (R1-2).} We thank the reviewer for this helpful comment on [topic]. We agree that [brief restatement of the concern], and we have revised the manuscript to [high-level description of what was done]. Specifically, we [added/clarified/extended] [methods/experiments/text], and we now [report/discuss] [key result]. These revisions aim to improve [clarity/robustness/transparency] as requested by the reviewer.
}
```

---

## 6. 修改说明（Change）——强制贴出关键文本

本规范要求：

- 对**所有实质性修改**（包括新增实验、重写段落、改图、改表）：
  - 必须在 `\revchange{...}` 中**贴出修改后的关键文本**；
  - 且必须指明**章节 + 页码 + 位置**；
  - 如涉及图表，还要说明图表编号和核心内容。

推荐模板：

```tex
\revchange{%
\textbf{Change (R1-2) in the manuscript.}
In Section~3.2 (p.~15), we revised the second paragraph to clarify [topic]. The updated text reads as follows:

\begin{quote}
[在此粘贴修改后的关键段落。可以略去与本评论无关的句子，但必须包含 reviewer 能够在稿件中定位的完整语境。]
\end{quote}

In addition, we updated Figure~3 (p.~18) to include [new elements], and we clarified the caption as follows:

\begin{quote}
[在此粘贴更新后的图注 caption 关键部分。]
\end{quote}

These changes ensure that [简要说明修改的效果，例如 “the role of PCA is clearly presented as a visualization tool rather than a predictive baseline”].
}
```

---

## 7. 评论编号与多子问题处理（R1-2.1 规范）

### 7.1 评论编号规则

- Associate Editor：
  - `AE-1`, `AE-2`, …；
- Reviewer #k：
  - `Rk-1`, `Rk-2`, …；
- 对于一个长评论中包含多个子问题，使用：
  - `Rk-2.1`, `Rk-2.2`, `Rk-2.3`，表示「评论 2 的子问题 1/2/3」。

在 LaTeX 中：

- 原始评论统一作为一个 `\revcomment{...}`，前缀写最大粒度编号，如：

  ```tex
  \revcomment{%
  \textbf{Comment R2-2.} [Reviewer 原文，可能包含多个子问题。]
  }
  ```

- 综合回复和修改说明中，内部用子编号标记，如：

  ```tex
  \revresponse{%
  \textbf{Response (R2-2.1).} [回复子问题 1]

  \medskip

  \textbf{Response (R2-2.2).} [回复子问题 2]

  \medskip

  \textbf{Response (R2-2.3).} [回复子问题 3]
  }
  ```

  ```tex
  \revchange{%
  \textbf{Change (R2-2.1) in the manuscript.}
  [对应子问题 1 的修改...]

  \medskip

  \textbf{Change (R2-2.2) in the manuscript.}
  [对应子问题 2 的修改...]

  \medskip

  \textbf{Change (R2-2.3) in the manuscript.}
  [对应子问题 3 的修改...]
  }
  ```

- 这样可以保证：
  - Reviewer 能清晰对应自己提的每一个子问题；
  - 将来如需自动解析 `response_to_reviewers.tex`，也有稳定的 ID 规则可用。

---

## 8. 内容定位与引用规范（页码、图表、参考文献）

### 8.1 页码与位置

- 页码以**最终提交 PDF 的页码**为准（干净稿，无标记版本）；
- 在 `\revchange` 中推荐格式：
  - `In Section~3.6 (p.~28), the last paragraph now reads as follows: ...`
  - 如需更精确，可加位置描述：
    - `the second paragraph`, `the last sentence`, `the first bullet`。

### 8.2 图表与补充材料

- 正文：
  - `Figure~n (p.~x)`，`Table~m (p.~y)`；
- 补充材料：
  - `Supplementary Figure~Sn (p.~xx)`, `Supplementary Table~Sm (p.~yy)`；
- 如图或表是**新增**的，建议在 `\revchange` 中明确：

  ```tex
  We added a new figure, Figure~13 (p.~29), to visualize [...]. The caption reads:

  \begin{quote}
  [粘贴 caption 关键内容]
  \end{quote}
  ```

### 8.3 参考文献引用

- 在综合回复中，如需引用已有工作：
  - 使用期刊要求的参考文献格式（如 natbib 的 `\citet`, `\citep`）；
- 如果本次修回**新增了参考文献**：
  - 在 `\revchange` 中说明：
    - 添加到 reference list；
    - 在正文具体哪里引用了这些文献；
  - 如现有范例那样，也可在 `\revchange` 中列出简要书目信息，方便 reviewers 查阅。

---

## 9. 完整示例

### 9.1 简单评论示例

```tex
\revcomment{%
\textbf{Comment R1-1.} The description of the main contribution in the Abstract appears too generic. Please clarify what is truly novel compared to existing deep learning approaches for [domain].
}

\revresponse{%
\textbf{Response (R1-1).} We thank the reviewer for this helpful suggestion. We agree that the original Abstract did not clearly distinguish our contributions from prior deep learning work. We have therefore revised the Abstract to explicitly highlight (i) the environment-specific architectural design, (ii) the integration of knowledge-guided feature selection, and (iii) the scale and robustness of the evaluation dataset.
}

\revchange{%
\textbf{Change (R1-1) in the manuscript.}
In the Abstract (p.~1), we revised the second and third sentences to better articulate the novel aspects of our framework. The updated text reads as follows:

\begin{quote}
[在此粘贴修改后的 Abstract 关键句子]
\end{quote}

These changes make the novelty of the work more explicit, as requested by the reviewer.
}
```

### 9.2 多子问题复杂评论示例

```tex
\revcomment{%
\textbf{Comment R2-2.} I am concerned about the clarity and reliability of detecting [X] across both terrestrial and marine environments, given their substantial differences. How do the authors substantiate the claim that such anomalies can be consistently observed in both settings? Additionally, the reference to a ``brightness glow'' in marine environments warrants clarification. Moreover, since oceanic events are frequently accompanied by [Y], to what extent can the authors rule out the possibility that the observed anomalies are [Y]-related rather than purely [X]-related?
}

\revresponse{%
\textbf{Response (R2-2.1).} We appreciate the reviewer’s careful scrutiny of cross-environment consistency. To substantiate the claim that [X] can be consistently observed across terrestrial and marine environments, we have added a dedicated cross-environment comparison in Section~3.4. This analysis quantifies [metrics] under matched conditions and confirms that the anomaly patterns are statistically consistent between land and ocean cases.

\medskip

\textbf{Response (R2-2.2).} We agree that the phrase ``brightness glow'' is potentially misleading. We have removed this wording from the manuscript and now consistently use the more precise term ``brightness temperature anomalies'' to describe the observed signals.

\medskip

\textbf{Response (R2-2.3).} Regarding the influence of [Y] (e.g., tsunamis), we have extended our control experiments by excluding [Y]-affected time windows and high-[Y]-state conditions in marine zones. The updated results, presented in Supplementary Figure~S2, show that the main conclusions remain unchanged, indicating that the reported anomalies are not artifacts of [Y].
}

\revchange{%
\textbf{Change (R2-2.1) in the manuscript.}
In Section~3.4 (p.~22), we added a new paragraph and a panel in Figure~7 to present a cross-environment comparison:

\begin{quote}
[粘贴新加的段落或其核心部分]
\end{quote}

Figure~7 (p.~23) now includes an additional panel (c) comparing [metrics] between terrestrial and marine environments.

\medskip

\textbf{Change (R2-2.2) in the manuscript.}
We removed the term ``brightness glow'' from the Introduction (p.~3) and Section~4.1 (p.~30). In both places, we now use the term ``brightness temperature anomalies'' instead.

\medskip

\textbf{Change (R2-2.3) in the manuscript.}
In Section~3.6 (p.~28), we added a subsection describing marine-zone controls that exclude [Y]-affected windows and high-[Y]-state days. The new paragraph reads:

\begin{quote}
[粘贴新加的控制实验描述]
\end{quote}

We also added Supplementary Figure~S2 (p.~65), which summarizes the results of these controls and shows that the key performance metrics remain stable after excluding [Y]-related periods.
}
```

---

## 10. 工作流程与最终检查清单

### 10.1 推荐工作流程

1. **收集评论**
   - 从 journal 系统或 PDF 中复制所有 AE/Reviewers 评论；
   - 定义编号：`AE-1`, `AE-2`, `R1-1`, `R1-2`, ..., 按原顺序排列。

2. **为每条评论创建骨架**
   - 在 `response_to_reviewers.tex` 中，对每条评论建立三段式框架：
     - `\revcomment{...}`
     - `\revresponse{...}`
     - `\revchange{...}`
   - 先填好 `\revcomment`（原文，一字不删）。

3. **制定修改方案并实施到主稿中**
   - 根据评论内容，先在主稿、图表、补充材料中完成修改；
   - 确定最终页码、图表编号。

4. **撰写 Response 和 Change**
   - `\revresponse`：写结构化、礼貌、数据支撑的综合回复；
   - `\revchange`：严格按照本规范指明：
     - 章节 + 页码（`p.~nn`）；
     - 修改后的关键文本（必须贴出）；
     - 涉及的图、表、Supplement 编号。

5. **全局一致性检查**
   - 确认**没有任何一条评论被遗漏**；
   - 每条评论的 Change 是否对应到主稿/补充材料中的真实改动；
   - 页码、图号、表号与最终 PDF 一致；
   - 颜色宏使用正确：评论蓝色、回复绿色、修改红色。

### 10.2 最终 Checklist（提交前）

- [ ] 每条 AE/Reviewer 评论都有对应的 `\revcomment{}` 块；
- [ ] 每个 `\revcomment{}` 后都有 `\revresponse{}` 和 `\revchange{}`；
- [ ] 所有实质修改在 `\revchange` 中**贴出了关键文本**；
- [ ] 所有位置引用都包含：章节 + 页码（`p.~nn`）；
- [ ] 图/表/补充材料编号在 `response_to_reviewers.tex` 与主稿中一致；
- [ ] 语言礼貌、准确，避免情绪化词汇；
- [ ] 所有新增参考文献都已加入 reference list 并在正文中被引用；
- [ ] 编译后的 `response_to_reviewers.pdf` 视觉上颜色区分明显，结构清晰。

