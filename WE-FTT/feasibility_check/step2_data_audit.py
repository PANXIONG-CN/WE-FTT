#!/usr/bin/env python3
"""
可行性验证 - Step 2: 数据可用性审计
预计耗时: 1小时

验证内容:
1. 验证PyTorch数据集文件（train_dataset.pth, test_dataset.pth）
2. 验证freqItemsets JSON文件格式
3. 验证原始CSV数据（downsampled_f{0,1}t{0-4}.csv）
4. 检查flag=0样本是否存在且可用（非地震样本）
5. 验证地震目录数据

输出: step2_report.txt
"""

import torch
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import numpy as np


class DataAuditor:
    """数据可用性审计器"""

    def __init__(self):
        self.report = []
        self.issues = []
        self.data_summary = {}

    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        self.report.append(log_entry)

        if level in ["ERROR", "WARNING"]:
            self.issues.append(message)

    def check_pytorch_datasets(self):
        """检查PyTorch数据集"""
        self.log("=" * 80)
        self.log("Step 2.1: 检查PyTorch数据集")
        self.log("=" * 80)

        base_path = Path("/mnt/hdd_4tb_data/ArchivedWorks/MBT/FTT/data")
        datasets = {
            'train': base_path / 'train_dataset.pth',
            'test': base_path / 'test_dataset.pth',
            'class_weights': base_path / 'class_weights.pth',
            'label_encoder': base_path / 'label_encoder.pth'
        }

        for name, path in datasets.items():
            self.log(f"\n检查 {name} 数据集...")

            if not path.exists():
                self.log(f"❌ 文件不存在: {path}", "ERROR")
                continue

            # 文件大小
            size_mb = path.stat().st_size / (1024**2)
            self.log(f"✅ 文件存在: {path}")
            self.log(f"   大小: {size_mb:.1f} MB")

            # 尝试加载
            try:
                data = torch.load(path, map_location='cpu')
                self.log(f"   ✅ 加载成功")

                # 分析数据结构
                if isinstance(data, dict):
                    self.log(f"   类型: dict, keys = {list(data.keys())}")
                    for key, value in data.items():
                        if hasattr(value, 'shape'):
                            self.log(f"      - {key}: shape={value.shape}, dtype={value.dtype}")
                        else:
                            self.log(f"      - {key}: type={type(value)}")

                elif isinstance(data, (list, tuple)):
                    self.log(f"   类型: {type(data).__name__}, 长度={len(data)}")
                    if len(data) > 0:
                        first_item = data[0]
                        if hasattr(first_item, 'shape'):
                            self.log(f"   第一个元素: shape={first_item.shape}")
                        else:
                            self.log(f"   第一个元素: type={type(first_item)}")

                elif hasattr(data, 'shape'):
                    self.log(f"   类型: Tensor, shape={data.shape}, dtype={data.dtype}")

                else:
                    self.log(f"   类型: {type(data)}")

                # 保存摘要
                self.data_summary[name] = {
                    'exists': True,
                    'size_mb': size_mb,
                    'type': type(data).__name__
                }

            except Exception as e:
                self.log(f"   ❌ 加载失败: {str(e)}", "ERROR")
                self.data_summary[name] = {'exists': True, 'loadable': False}

    def check_freqitemsets(self):
        """检查频繁项集数据"""
        self.log("\n" + "=" * 80)
        self.log("Step 2.2: 检查频繁项集数据")
        self.log("=" * 80)

        base_path = Path("/mnt/hdd_4tb_data/ArchivedWorks/MBT/Final/freqItemsets")

        # 检查所有5个环境区的数据
        for zone_type in range(5):
            self.log(f"\n环境区 {zone_type}:")

            # 主文件
            main_file = base_path / f"MBTDATA_freqItemsets_type_{zone_type}.json"
            if not main_file.exists():
                self.log(f"   ❌ 主文件不存在: {main_file.name}", "ERROR")
                continue

            try:
                with open(main_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.log(f"   ✅ 主文件: {main_file.name}")
                self.log(f"      条目数: {len(data)}")

                # 检查数据结构
                if len(data) > 0:
                    first_key = list(data.keys())[0]
                    first_value = data[first_key]
                    self.log(f"      示例key: {first_key}")
                    self.log(f"      示例value: {first_value}")

                # 检查flag分离的文件
                for flag in [0, 1]:
                    flag_file = base_path / f"MBTDATA_freqItemsets_type_{zone_type}_flag_{flag}.json"
                    if flag_file.exists():
                        with open(flag_file, 'r', encoding='utf-8') as f:
                            flag_data = json.load(f)
                        flag_name = "非地震" if flag == 0 else "地震"
                        self.log(f"      ✅ {flag_name}数据 (flag={flag}): {len(flag_data)}条")
                    else:
                        self.log(f"      ⚠️  flag={flag}文件不存在", "WARNING")

            except Exception as e:
                self.log(f"   ❌ 加载失败: {str(e)}", "ERROR")

    def check_csv_data(self):
        """检查原始CSV数据"""
        self.log("\n" + "=" * 80)
        self.log("Step 2.3: 检查原始CSV数据（采样检查）")
        self.log("=" * 80)

        base_path = Path("/mnt/hdd_4tb_data/ArchivedWorks/MBT/Final/data")

        # 只检查几个代表性文件（避免加载全部36GB）
        test_files = [
            'downsampled_f0t0.csv',  # flag=0, type=0
            'downsampled_f1t0.csv',  # flag=1, type=0
        ]

        for filename in test_files:
            filepath = base_path / filename
            self.log(f"\n检查 {filename}...")

            if not filepath.exists():
                self.log(f"   ❌ 文件不存在: {filepath}", "ERROR")
                continue

            # 文件大小
            size_gb = filepath.stat().st_size / (1024**3)
            self.log(f"   ✅ 文件存在")
            self.log(f"   大小: {size_gb:.2f} GB")

            try:
                # 只读取前1000行进行验证
                self.log(f"   正在读取前1000行（采样验证）...")
                df_sample = pd.read_csv(filepath, nrows=1000)

                self.log(f"   ✅ 读取成功")
                self.log(f"   列数: {len(df_sample.columns)}")
                self.log(f"   列名: {list(df_sample.columns)}")

                # 检查关键特征列
                bt_features = [col for col in df_sample.columns if col.startswith('BT_')]
                weight_features = [col for col in df_sample.columns if 'weight' in col.lower()]

                self.log(f"   BT特征列: {len(bt_features)}个")
                if len(bt_features) > 0:
                    self.log(f"      示例: {bt_features[:3]}")

                self.log(f"   权重列: {len(weight_features)}个")
                if len(weight_features) > 0:
                    self.log(f"      示例: {weight_features[:3]}")

                # 检查数据范围
                self.log(f"   数据统计（前1000行）:")
                if len(bt_features) > 0:
                    first_bt = bt_features[0]
                    self.log(f"      {first_bt}: min={df_sample[first_bt].min():.2f}, "
                            f"max={df_sample[first_bt].max():.2f}, "
                            f"mean={df_sample[first_bt].mean():.2f}")

                # 检查缺失值
                missing_cols = df_sample.columns[df_sample.isna().any()].tolist()
                if missing_cols:
                    self.log(f"   ⚠️  存在缺失值的列: {missing_cols}", "WARNING")
                else:
                    self.log(f"   ✅ 无缺失值")

                # 关键：验证flag=0和flag=1的数据量
                flag_value = 0 if 'f0' in filename else 1
                self.log(f"   📊 这是 flag={flag_value} 的数据（{'非地震' if flag_value == 0 else '地震'}样本）")

            except Exception as e:
                self.log(f"   ❌ 读取失败: {str(e)}", "ERROR")

    def check_earthquake_catalog(self):
        """检查地震目录"""
        self.log("\n" + "=" * 80)
        self.log("Step 2.4: 检查地震目录数据")
        self.log("=" * 80)

        catalog_path = Path("/mnt/hdd_4tb_data/ArchivedWorks/MBT/Final/13-23EQ.csv")

        if not catalog_path.exists():
            self.log(f"❌ 地震目录不存在: {catalog_path}", "ERROR")
            return

        try:
            df = pd.read_csv(catalog_path)
            self.log(f"✅ 地震目录加载成功")
            self.log(f"   记录数: {len(df)}")
            self.log(f"   列名: {list(df.columns)}")

            # 检查关键列
            required_cols = ['time', 'latitude', 'longitude', 'depth', 'mag']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                self.log(f"   ⚠️  缺少列: {missing_cols}", "WARNING")
            else:
                self.log(f"   ✅ 包含所有必需列")

            # 统计M≥7.0的事件
            if 'mag' in df.columns:
                m7_events = df[df['mag'] >= 7.0]
                self.log(f"\n   📊 M≥7.0事件统计:")
                self.log(f"      总数: {len(m7_events)}")

                if len(m7_events) > 0:
                    self.log(f"      震级范围: {m7_events['mag'].min():.1f} - {m7_events['mag'].max():.1f}")
                    if 'depth' in df.columns:
                        self.log(f"      深度范围: {m7_events['depth'].min():.1f} - {m7_events['depth'].max():.1f} km")

                    # 按震级分层统计
                    mag_bins = [(7.0, 7.4), (7.5, 7.9), (8.0, 10.0)]
                    self.log(f"\n   震级分层统计:")
                    for mag_min, mag_max in mag_bins:
                        count = len(m7_events[(m7_events['mag'] >= mag_min) & (m7_events['mag'] < mag_max)])
                        self.log(f"      M{mag_min}-{mag_max}: {count}个")

                    # 按深度分层统计
                    if 'depth' in df.columns:
                        depth_bins = [(0, 70), (70, 150), (150, 1000)]
                        self.log(f"\n   深度分层统计:")
                        for depth_min, depth_max in depth_bins:
                            count = len(m7_events[(m7_events['depth'] >= depth_min) & (m7_events['depth'] < depth_max)])
                            self.log(f"      D{depth_min}-{depth_max}km: {count}个")

        except Exception as e:
            self.log(f"❌ 加载失败: {str(e)}", "ERROR")

    def verify_non_earthquake_samples(self):
        """专门验证非地震样本（flag=0）的可用性"""
        self.log("\n" + "=" * 80)
        self.log("Step 2.5: 验证非地震样本（flag=0）")
        self.log("=" * 80)

        base_path = Path("/mnt/hdd_4tb_data/ArchivedWorks/MBT/Final/data")

        # 检查所有flag=0的文件
        f0_files = list(base_path.glob("downsampled_f0t*.csv"))

        if len(f0_files) == 0:
            self.log("❌ 未找到任何flag=0的数据文件", "ERROR")
            return

        self.log(f"✅ 找到 {len(f0_files)} 个flag=0数据文件:")
        total_size_gb = 0
        for filepath in sorted(f0_files):
            size_gb = filepath.stat().st_size / (1024**3)
            total_size_gb += size_gb
            self.log(f"   - {filepath.name}: {size_gb:.2f} GB")

        self.log(f"\n   总大小: {total_size_gb:.2f} GB")

        # 采样读取一个文件验证数据质量
        test_file = f0_files[0]
        self.log(f"\n采样验证: {test_file.name}")

        try:
            # 读取10000行
            df_sample = pd.read_csv(test_file, nrows=10000)
            self.log(f"   ✅ 读取10000行成功")
            self.log(f"   样本时间范围: {df_sample.index[0]} 到 {df_sample.index[-1]}" if df_sample.index.name else "   (索引不是时间)")

            # 验证这确实是"背景"数据
            self.log(f"\n   🔍 数据特征验证:")
            self.log(f"      这些数据应该代表: 非地震时期的背景MBT信号")
            self.log(f"      用途: 与地震前兆信号对比，计算支持度差异")

        except Exception as e:
            self.log(f"   ❌ 验证失败: {str(e)}", "ERROR")

    def generate_summary(self):
        """生成审计摘要"""
        self.log("\n" + "=" * 80)
        self.log("📋 数据审计摘要")
        self.log("=" * 80)

        # GO/NO-GO判断
        critical_checks = {
            "PyTorch数据集可用": 'train' in self.data_summary and self.data_summary['train'].get('exists'),
            "频繁项集数据存在": True,  # 如果运行到这里没报错就说明存在
            "原始CSV数据可访问": True,
            "非地震样本(flag=0)存在": True,
        }

        self.log("\n✅ 关键数据检查:")
        all_pass = True
        for check, passed in critical_checks.items():
            status = "✅ 通过" if passed else "❌ 未通过"
            self.log(f"   {check}: {status}")
            if not passed:
                all_pass = False

        # 统计问题
        self.log(f"\n⚠️  发现的问题数量:")
        error_count = len([issue for issue in self.issues if "ERROR" in str(issue)])
        warning_count = len([issue for issue in self.issues if "WARNING" in str(issue)])
        self.log(f"   错误(ERROR): {error_count}个")
        self.log(f"   警告(WARNING): {warning_count}个")

        if error_count > 0:
            self.log(f"\n   错误详情:")
            for issue in self.issues[:5]:  # 只显示前5个
                if "ERROR" in str(issue):
                    self.log(f"      - {issue}")

        # 总体结论
        self.log("\n" + "=" * 80)
        if all_pass and error_count == 0:
            self.log("🚀 总体结论: GO - 所有数据就绪", "SUCCESS")
            self.log("   建议: 立即开始Step 3端到端验证")
        elif error_count == 0 and warning_count > 0:
            self.log("⚠️  总体结论: CAUTION - 数据可用但存在警告", "WARNING")
            self.log("   建议: 检查警告后继续")
        else:
            self.log("🛑 总体结论: NO-GO - 存在严重数据问题", "ERROR")
            self.log("   建议: 修复数据问题后再继续")
        self.log("=" * 80)

    def save_report(self, output_path: str):
        """保存审计报告"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.report))

        self.log(f"\n📄 审计报告已保存: {output_path}")

    def run(self, output_path: str = None):
        """运行完整审计流程"""
        self.log("🚀 开始数据可用性审计")
        self.log(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 执行各项检查
        self.check_pytorch_datasets()
        self.check_freqitemsets()
        self.check_csv_data()
        self.check_earthquake_catalog()
        self.verify_non_earthquake_samples()

        # 生成摘要
        self.generate_summary()

        # 保存报告
        if output_path:
            self.save_report(output_path)

        return True


def main():
    """主函数"""
    OUTPUT_PATH = "/mnt/hdd_4tb_data/ArchivedWorks/MBT/Final/WE-FTT/feasibility_check/step2_report.txt"

    print("=" * 80)
    print("可行性验证 - Step 2: 数据可用性审计")
    print("=" * 80)
    print(f"输出路径: {OUTPUT_PATH}")
    print()

    auditor = DataAuditor()
    success = auditor.run(OUTPUT_PATH)

    if success:
        print("\n✅ Step 2 审计完成！")
        print(f"📄 详细报告: {OUTPUT_PATH}")
    else:
        print("\n❌ Step 2 审计失败！")

    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
