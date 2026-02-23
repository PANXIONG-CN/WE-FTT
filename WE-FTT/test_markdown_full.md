# Markdown 预览测试文档

## 基础功能测试

### 文本格式
- **粗体文字**
- *斜体文字*
- ~~删除线~~
- `行内代码`

### 列表
1. 有序列表项 1
2. 有序列表项 2
   - 嵌套无序列表
   - 另一个嵌套项

### 引用
> 这是一个引用块
> 可以有多行

### 代码块
```python
def hello_world():
    """测试函数"""
    print("Hello, World!")
    return True
```

### 表格
| 功能 | 状态 | 说明 |
|------|------|------|
| pandoc | ✅ 已安装 | v3.1.13 |
| pandoc-crossref | ✅ 已安装 | v0.3.17.0 |
| Markdown Preview Enhanced | ✅ 已配置 | 使用 pandoc 解析器 |

### 链接和图片
[GitHub](https://github.com)

### 数学公式（KaTeX）
行内公式：$E = mc^2$

块级公式：
$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$

### 任务列表
- [x] 安装 pandoc
- [x] 安装 pandoc-crossref
- [x] 配置 VS Code
- [ ] 享受 Markdown 预览

### HTML 元素
<kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>V</kbd>

### 水平线
---

## 总结
如果您能看到以上所有元素正确渲染，说明 Markdown 预览功能已经完全正常工作！