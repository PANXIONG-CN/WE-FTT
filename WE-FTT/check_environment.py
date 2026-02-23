#!/usr/bin/env python3
"""
环境检查脚本 - 验证双GPU训练环境
"""

import sys
import os

def check_environment():
    print("=" * 60)
    print("WE-FTT 双GPU训练环境检查")
    print("=" * 60)
    print()

    # 1. Python版本
    print(f"✓ Python版本: {sys.version}")
    print()

    # 2. PyTorch
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"✓ CUDA版本: {torch.version.cuda}")
            print(f"✓ cuDNN版本: {torch.backends.cudnn.version()}")
            print(f"✓ GPU数量: {torch.cuda.device_count()}")
            print()

            # 列出所有GPU
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    总显存: {props.total_memory / 1024**3:.1f} GB")
                print(f"    计算能力: {props.major}.{props.minor}")
                print()

            # 检查NVLink
            if torch.cuda.device_count() >= 2:
                print("检查NVLink连接...")
                try:
                    # 尝试在两个GPU之间传输数据
                    device0 = torch.device('cuda:0')
                    device1 = torch.device('cuda:1')

                    # 创建测试张量
                    x = torch.randn(1000, 1000, device=device0)
                    y = x.to(device1)

                    print("✓ GPU 0 <-> GPU 1 数据传输正常")
                    print()
                except Exception as e:
                    print(f"✗ GPU间数据传输失败: {e}")
                    print()
        else:
            print("✗ CUDA不可用!")
            print()
    except ImportError:
        print("✗ PyTorch未安装!")
        print()
        return False

    # 3. 检查必需的库
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'scipy',
        'matplotlib', 'seaborn', 'tqdm', 'yaml',
        'pyarrow'
    ]

    print("检查必需的包:")
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (未安装)")
            all_installed = False
    print()

    # 4. 检查数据
    print("检查数据文件:")
    data_dir = "data/processed"
    if os.path.exists(data_dir):
        print(f"  ✓ 数据目录存在: {data_dir}")

        # 列出parquet文件
        import glob
        parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
        if parquet_files:
            print(f"  ✓ 找到 {len(parquet_files)} 个parquet文件")
            for f in parquet_files[:5]:  # 只显示前5个
                print(f"    - {os.path.basename(f)}")
        else:
            print("  ✗ 未找到parquet文件")
    else:
        print(f"  ✗ 数据目录不存在: {data_dir}")
    print()

    # 5. 测试分布式训练
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print("测试分布式训练设置...")
        try:
            import torch.distributed as dist
            print("  ✓ torch.distributed 可用")

            # 检查NCCL后端
            if dist.is_nccl_available():
                print("  ✓ NCCL后端可用")
            else:
                print("  ✗ NCCL后端不可用")
        except Exception as e:
            print(f"  ✗ 分布式训练检查失败: {e}")
        print()

    # 6. 内存检查
    print("GPU显存检查:")
    if torch.cuda.is_available():
        for i in range(min(2, torch.cuda.device_count())):
            torch.cuda.set_device(i)
            free_mem = torch.cuda.get_device_properties(i).total_memory
            allocated = torch.cuda.memory_allocated(i)
            cached = torch.cuda.memory_reserved(i)

            print(f"  GPU {i}:")
            print(f"    总显存: {free_mem / 1024**3:.1f} GB")
            print(f"    已分配: {allocated / 1024**3:.3f} GB")
            print(f"    缓存: {cached / 1024**3:.3f} GB")
            print(f"    可用: {(free_mem - allocated) / 1024**3:.1f} GB")
    print()

    print("=" * 60)
    if all_installed and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print("✓ 环境检查通过! 可以开始训练。")
    else:
        print("✗ 环境检查未完全通过，请检查上述问题。")
    print("=" * 60)

    return True


if __name__ == '__main__':
    check_environment()
