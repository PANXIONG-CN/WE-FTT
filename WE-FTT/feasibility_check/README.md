# 可行性验证脚本包

**目的**: 在投入大量时间实施论文返修补充实验前，快速验证关键资源和流程的可行性

**预计总耗时**: 3.5小时

---

## 📋 验证内容

### Step 1: 模型快速验证（30分钟）
验证内容:
- ✅ 模型checkpoint文件存在且可加载
- ✅ 判断模型类型（WE-FTT vs FT-Transformer）
- ✅ 检查性能指标是否匹配论文（MCC≈0.84）
- ✅ 验证模型包含推理所需组件

输出: `step1_report.txt`

### Step 2: 数据可用性审计（1小时）
验证内容:
- ✅ PyTorch数据集文件（train_dataset.pth, test_dataset.pth）
- ✅ 频繁项集JSON文件（地震和非地震样本）
- ✅ 原始CSV数据（36GB）
- ✅ 地震目录数据
- ✅ flag=0非地震样本存在性

输出: `step2_report.txt`

### Step 3: 端到端最小化验证（2小时）
验证内容:
- ✅ 单个地震事件的模型推理流程
- ✅ 单个环境区的物理机制统计分析
- ✅ 数据加载和筛选逻辑
- ✅ Bootstrap置信区间计算
- ✅ 性能评估（耗时统计）

输出: `step3_report.txt`

### 最终决策报告
基于前3步的结果，自动生成GO/NO-GO决策

输出: `FINAL_DECISION.txt`

---

## 🚀 使用方法

### 方法1: 一键运行所有验证（推荐）

```bash
cd /mnt/hdd_4tb_data/ArchivedWorks/MBT/Final/WE-FTT/feasibility_check
python run_all_checks.py
```

这将自动依次执行Step 1、Step 2、Step 3，并生成最终决策报告。

### 方法2: 单独运行各步骤

如果你想分步执行并查看每步结果：

```bash
# Step 1: 模型验证
python step1_model_verification.py

# 查看报告
cat step1_report.txt

# Step 2: 数据审计
python step2_data_audit.py

# 查看报告
cat step2_report.txt

# Step 3: 端到端验证
python step3_end_to_end.py

# 查看报告
cat step3_report.txt
```

---

## 📊 输出文件

运行完成后，会生成以下文件：

```
feasibility_check/
├── step1_report.txt           # 模型验证详细报告
├── step2_report.txt           # 数据审计详细报告
├── step3_report.txt           # 端到端验证详细报告
├── FINAL_DECISION.txt         # 最终GO/NO-GO决策
└── README.md                  # 本说明文档
```

---

## 🎯 决策矩阵

### GO - 可以立即开始

**条件**:
- ✅ 模型是WE-FTT
- ✅ 模型性能匹配论文（MCC≈0.84）
- ✅ 所有数据可用
- ✅ 端到端流程可行

**行动**:
1. 立即开始物理机制统计分析（优先级P1）
2. 并行开始震源学分层评估（优先级P1）
3. 按照EXECUTION_PLAN.md推进

### CAUTION - 需要调整但可以继续

**情况1: 模型是FT-Transformer**
- ⚠️ 找到的是基线模型而非WE-FTT
- **解决方案**:
  1. 继续搜索WE-FTT模型
  2. 或使用FT-Transformer并在论文中说明
  3. 最后方案：重新训练（1-2天）

**情况2: 数据有警告但可用**
- ⚠️ 存在缺失值或格式问题
- **解决方案**: 记录问题，在实施中处理

### NO-GO - 存在阻塞问题

**情况1: 数据缺失**
- ❌ 关键数据文件不存在
- **解决方案**: 找到数据或重新生成

**情况2: 端到端流程失败**
- ❌ 多个关键错误
- **解决方案**: 修复技术问题后重新验证

---

## ⚠️ 常见问题

### Q1: 报告中出现"WARNING"怎么办？
A: WARNING不会阻止继续，但需要注意。查看详细报告确认问题，在实施中处理。

### Q2: 某个步骤失败了怎么办？
A:
1. 查看对应的report.txt找到详细错误
2. 如果是路径问题，检查文件是否存在
3. 如果是权限问题，检查文件权限
4. 修复后可以单独重新运行该步骤

### Q3: 验证通过后，下一步做什么？
A: 查看FINAL_DECISION.txt中的"执行建议"，按照EXECUTION_PLAN.md开始实施。

### Q4: 需要GPU吗？
A: 不需要。可行性验证只加载模型检查架构，不做实际训练或大规模推理。

---

## 🔧 技术细节

### 依赖项
- Python 3.7+
- PyTorch
- pandas
- numpy
- json (标准库)

### 资源需求
- 内存: 约2-4GB（加载checkpoint和少量数据）
- 磁盘: 只读取，不写入大文件
- 时间: 3.5小时（保守估计）

### 安全性
- ✅ 只读操作，不修改任何原始数据
- ✅ 不训练模型，不消耗GPU资源
- ✅ 可以安全地重复运行

---

## 📞 问题反馈

如果验证过程中遇到未预料的问题，请记录：
1. 错误信息（完整的traceback）
2. 你的环境（Python版本、操作系统）
3. 报告文件内容

---

**创建时间**: 2025-11-04
**作者**: Claude Code
**版本**: v1.0
