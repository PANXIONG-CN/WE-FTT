# PyGMT配置指南

## 问题背景

PyGMT 需要底层的 GMT（Generic Mapping Tools）库支持。由于C++标准库版本依赖，在混合使用 conda 和 uv 环境时可能出现库加载错误。

## 已完成的配置

1. ✅ **GMT 6.6.0** 已通过 conda 安装
2. ✅ **PyGMT 0.17.0** 已安装到 conda 环境
3. ✅ **SQLite** 已更新以支持 GDAL 依赖
4. ✅ **包装脚本** 已创建 (`run_with_gmt.sh`)

## 使用方法

### 方法1：使用包装脚本（推荐）

```bash
# 运行需要PyGMT的脚本
./run_with_gmt.sh supplement_experiments/exp2/scripts/generate_fig_s2.py

# 或者使用相对路径
./run_with_gmt.sh scripts/train.py --model we_ftt
```

### 方法2：手动设置环境变量

```bash
# 单次运行
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" python your_script.py

# 或在当前shell会话中永久设置
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python your_script.py
```

### 方法3：添加到 shell 配置（全局永久）

在 `~/.bashrc` 或 `~/.zshrc` 中添加：

```bash
# PyGMT库路径（仅在gpytorch环境激活时有效）
if [[ "$CONDA_DEFAULT_ENV" == "gpytorch" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi
```

然后重新加载配置：
```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

## 验证安装

测试 PyGMT 是否正常工作：

```bash
./run_with_gmt.sh -c "import pygmt; print('PyGMT', pygmt.__version__, 'OK'); pygmt.show_versions()"
```

预期输出应显示 PyGMT 版本和 GMT 6.6.0 信息。

## uv环境说明

**重要**：`uv run` 不适用于 PyGMT，因为：
1. uv 创建的虚拟环境无法访问 conda 的系统库（如 GMT、GDAL）
2. C++ ABI 版本冲突无法通过 Python 包管理器解决

**建议**：
- 需要 PyGMT 的脚本：使用 `python` (conda环境) 或 `./run_with_gmt.sh`
- 不需要 PyGMT 的脚本：可以继续使用 `uv run`

## 故障排查

### 错误1：`libstdc++.so.6: version 'CXXABI_1.3.15' not found`

**原因**：系统 C++ 标准库版本太旧
**解决**：使用 `run_with_gmt.sh` 或手动设置 `LD_LIBRARY_PATH`

### 错误2：`undefined symbol: sqlite3_total_changes64`

**原因**：SQLite 版本不匹配
**解决**：已通过 `conda update libsqlite sqlite` 修复

### 错误3：`No module named 'pygmt'`

**原因**：在 uv 虚拟环境中运行
**解决**：使用 conda 环境的 Python：`python your_script.py`

### 错误4：`Unrecognized parameter 'color'`

**原因**：PyGMT API 变化（v0.17.0 使用 `fill` 代替 `color`）
**解决**：已修复 `generate_fig_s2.py` 脚本

## 相关文件

- `run_with_gmt.sh`：PyGMT 运行包装脚本
- `requirements.txt`：包含 `pygmt>=0.10.0` 依赖
- `supplement_experiments/exp2/scripts/generate_fig_s2.py`：使用PyGMT的示例

## 技术细节

### GMT 版本信息
```
GMT: 6.6.0
PyGMT: 0.17.0
GDAL: 3.10.3
Ghostscript: 10.06.0
```

### 关键依赖链
```
PyGMT → GMT → GDAL → SQLite, libstdc++
```

### 库路径优先级
```
$CONDA_PREFIX/lib  (conda环境，优先)
    ↓
/usr/lib           (系统库，备选)
```

---

**最后更新**：2025-11-12
**适用环境**：conda gpytorch 环境 + Python 3.11.8
**GMT版本**：6.6.0
**PyGMT版本**：0.17.0
