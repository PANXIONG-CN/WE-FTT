---
description: 编译LaTeX文档,PDF保存在源文件目录,中间文件保存在build目录
---

请按照以下流程编译LaTeX文档:

## 参数
- `<tex_file>`: LaTeX源文件的完整路径(必需)

## 编译步骤

1. 验证源文件存在,获取文件目录和基础文件名
2. 进入源文件所在目录
3. 创建build子目录(如果不存在): `mkdir -p build`
4. 第一次编译: `pdflatex -output-directory=build -interaction=nonstopmode <文件名>.tex`
5. 处理参考文献: `bibtex build/<文件名>`
6. 第二次编译: `pdflatex -output-directory=build -interaction=nonstopmode <文件名>.tex`
7. 第三次编译: `pdflatex -output-directory=build -interaction=nonstopmode <文件名>.tex`
8. 复制PDF到主目录: `cp build/<文件名>.pdf ./<文件名>.pdf`
9. 删除build中的PDF: `rm build/<文件名>.pdf`
10. 显示结果:
    - 主目录中的PDF文件信息
    - build目录中的中间文件列表

## 输出结构
```
源文件目录/
├── <文件名>.tex    # LaTeX源文件
├── <文件名>.pdf    # 最终PDF(仅此文件)
└── build/          # 所有中间文件
    ├── *.aux
    ├── *.bbl
    ├── *.blg
    ├── *.log
    ├── *.out
    └── ...
```

## 注意事项
- 所有pdflatex命令使用`-interaction=nonstopmode`自动处理错误
- 参考文献文件(.bib)应在源文件同目录
- 如有图片,确保LaTeX中graphicspath正确设置
- 最终只保留PDF在主目录,所有中间文件在build目录
