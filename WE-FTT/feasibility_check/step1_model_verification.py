#!/usr/bin/env python3
"""
可行性验证 - Step 1: 模型快速验证
预计耗时: 30分钟

验证内容:
1. 加载checkpoint，检查结构
2. 判断是WE-FTT还是FT-Transformer
3. 检查性能指标是否匹配论文
4. 验证模型可以加载并准备推理

输出: step1_report.txt
"""

import torch
import json
from pathlib import Path
from datetime import datetime


class ModelVerifier:
    """模型验证器"""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.report = []
        self.is_we_ftt = None
        self.performance_match = None

    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        self.report.append(log_entry)

    def verify_file_exists(self) -> bool:
        """验证模型文件是否存在"""
        self.log("=" * 80)
        self.log("Step 1.1: 验证模型文件")
        self.log("=" * 80)

        if not self.model_path.exists():
            self.log(f"❌ 模型文件不存在: {self.model_path}", "ERROR")
            return False

        file_size_mb = self.model_path.stat().st_size / (1024**2)
        self.log(f"✅ 模型文件存在: {self.model_path}")
        self.log(f"   文件大小: {file_size_mb:.1f} MB")

        return True

    def load_checkpoint(self):
        """加载checkpoint"""
        self.log("\n" + "=" * 80)
        self.log("Step 1.2: 加载Checkpoint")
        self.log("=" * 80)

        try:
            self.log("正在加载checkpoint (可能需要几秒)...")
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.log("✅ Checkpoint加载成功")

            # 检查checkpoint的顶层keys
            self.log(f"\n📋 Checkpoint顶层keys ({len(checkpoint.keys())}个):")
            for key in sorted(checkpoint.keys()):
                self.log(f"   - {key}")

            return checkpoint

        except Exception as e:
            self.log(f"❌ 加载失败: {str(e)}", "ERROR")
            return None

    def analyze_model_architecture(self, checkpoint: dict) -> bool:
        """分析模型架构"""
        self.log("\n" + "=" * 80)
        self.log("Step 1.3: 分析模型架构")
        self.log("=" * 80)

        if 'model_state_dict' not in checkpoint:
            self.log("⚠️  checkpoint中没有'model_state_dict'", "WARNING")
            self.log(f"   可用的keys: {list(checkpoint.keys())}")
            return False

        model_state = checkpoint['model_state_dict']
        all_keys = list(model_state.keys())

        self.log(f"\n模型参数总数: {len(all_keys)}")
        self.log(f"\n前10个参数:")
        for key in all_keys[:10]:
            shape = model_state[key].shape if hasattr(model_state[key], 'shape') else 'N/A'
            self.log(f"   - {key}: {shape}")

        # 检查WE-FTT特征标志
        self.log("\n🔍 检查WE-FTT特征标志...")

        # 1. Weight projection层
        weight_proj_keys = [k for k in all_keys if 'weight_proj' in k.lower()]
        self.log(f"\n1. Weight Projection层: {len(weight_proj_keys)}个")
        if weight_proj_keys:
            for key in weight_proj_keys[:5]:
                self.log(f"   ✓ {key}")
        else:
            self.log(f"   ✗ 未找到")

        # 2. Feature projection层
        feature_proj_keys = [k for k in all_keys if 'feature_proj' in k.lower()]
        self.log(f"\n2. Feature Projection层: {len(feature_proj_keys)}个")
        if feature_proj_keys:
            for key in feature_proj_keys[:5]:
                self.log(f"   ✓ {key}")
        else:
            self.log(f"   ✗ 未找到")

        # 3. 其他可能的weight相关层
        other_weight_keys = [k for k in all_keys if 'weight' in k.lower() and 'weight_proj' not in k.lower()]
        self.log(f"\n3. 其他weight相关层: {len(other_weight_keys)}个")
        if len(other_weight_keys) > 0:
            self.log(f"   示例前3个:")
            for key in other_weight_keys[:3]:
                self.log(f"   - {key}")

        # 判断模型类型
        self.log("\n" + "=" * 80)
        if len(weight_proj_keys) > 0:
            self.log("🎯 判断结果: 这是 WE-FTT 模型 ✅")
            self.log(f"   依据: 存在 {len(weight_proj_keys)} 个weight_proj层")
            self.is_we_ftt = True
        elif len(feature_proj_keys) > 0:
            self.log("⚠️  判断结果: 这可能是 FT-Transformer (有feature_proj但无weight_proj)")
            self.is_we_ftt = False
        else:
            self.log("❓ 判断结果: 模型类型不明确")
            self.log("   建议: 手动检查模型架构代码")
            self.is_we_ftt = None
        self.log("=" * 80)

        return True

    def check_performance_metrics(self, checkpoint: dict):
        """检查性能指标"""
        self.log("\n" + "=" * 80)
        self.log("Step 1.4: 检查性能指标")
        self.log("=" * 80)

        # 论文目标性能
        target_mcc = 0.84
        tolerance = 0.05  # 允许±5%的偏差

        # 检查可能的性能字段
        perf_keys = ['best_mcc', 'mcc', 'best_val_mcc', 'val_mcc', 'test_mcc']
        found_mcc = None

        for key in perf_keys:
            if key in checkpoint:
                found_mcc = checkpoint[key]
                self.log(f"✅ 找到性能指标: {key} = {found_mcc:.4f}")
                break

        if found_mcc is None:
            self.log("⚠️  未找到MCC指标", "WARNING")
            self.log(f"   checkpoint中的数值型keys:")
            for key, value in checkpoint.items():
                if isinstance(value, (int, float)):
                    self.log(f"   - {key}: {value}")
            self.performance_match = None
        else:
            # 比较性能
            diff = abs(found_mcc - target_mcc)
            self.log(f"\n📊 性能比较:")
            self.log(f"   论文报告MCC: {target_mcc:.4f}")
            self.log(f"   模型实际MCC: {found_mcc:.4f}")
            self.log(f"   差异: {diff:.4f}")

            if diff <= tolerance:
                self.log(f"   ✅ 性能匹配（差异 ≤ {tolerance}）")
                self.performance_match = True
            else:
                self.log(f"   ⚠️  性能偏差较大（差异 > {tolerance}）", "WARNING")
                self.performance_match = False

        # 检查其他训练信息
        self.log(f"\n📝 其他训练信息:")
        info_keys = ['epoch', 'best_epoch', 'optimizer', 'lr', 'batch_size']
        for key in info_keys:
            if key in checkpoint:
                self.log(f"   - {key}: {checkpoint[key]}")

    def check_model_loadable(self, checkpoint: dict) -> bool:
        """检查模型是否可以用于推理"""
        self.log("\n" + "=" * 80)
        self.log("Step 1.5: 检查模型可加载性")
        self.log("=" * 80)

        # 必需的组件
        required_keys = ['model_state_dict']
        optional_keys = ['optimizer_state_dict', 'scheduler_state_dict']

        self.log("必需组件:")
        all_required = True
        for key in required_keys:
            if key in checkpoint:
                self.log(f"   ✅ {key}")
            else:
                self.log(f"   ❌ {key} (缺失)", "ERROR")
                all_required = False

        self.log("\n可选组件:")
        for key in optional_keys:
            if key in checkpoint:
                self.log(f"   ✅ {key}")
            else:
                self.log(f"   - {key} (未保存，不影响推理)")

        if all_required:
            self.log("\n✅ 模型包含推理所需的全部组件")
        else:
            self.log("\n❌ 模型缺少必需组件，可能无法直接使用", "ERROR")

        return all_required

    def generate_summary(self):
        """生成验证摘要"""
        self.log("\n" + "=" * 80)
        self.log("📋 验证摘要")
        self.log("=" * 80)

        # 判断GO/NO-GO
        go_criteria = {
            "模型类型是WE-FTT": self.is_we_ftt == True,
            "性能指标匹配论文": self.performance_match != False,  # None也可以接受
            "模型可加载": True  # 如果走到这里说明加载成功了
        }

        self.log("\n✅ GO条件检查:")
        all_go = True
        for criterion, passed in go_criteria.items():
            status = "✅ 通过" if passed else "❌ 未通过"
            self.log(f"   {criterion}: {status}")
            if not passed:
                all_go = False

        self.log("\n" + "=" * 80)
        if all_go:
            self.log("🚀 总体结论: GO - 可以继续执行后续实验", "SUCCESS")
            self.log("   建议: 立即开始Step 2数据审计")
        elif self.is_we_ftt == False:
            self.log("⚠️  总体结论: CAUTION - 找到的是FT-Transformer", "WARNING")
            self.log("   建议: 继续搜索WE-FTT模型，或使用此模型但需在论文中说明")
        else:
            self.log("🛑 总体结论: NO-GO - 存在严重问题", "ERROR")
            self.log("   建议: 检查模型文件或考虑重新训练")
        self.log("=" * 80)

    def save_report(self, output_path: str):
        """保存验证报告"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.report))

        self.log(f"\n📄 验证报告已保存: {output_path}")

    def run(self, output_path: str = None):
        """运行完整验证流程"""
        self.log("🚀 开始模型可行性验证")
        self.log(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 验证文件
        if not self.verify_file_exists():
            return False

        # 加载checkpoint
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return False

        # 分析架构
        if not self.analyze_model_architecture(checkpoint):
            return False

        # 检查性能
        self.check_performance_metrics(checkpoint)

        # 检查可加载性
        self.check_model_loadable(checkpoint)

        # 生成摘要
        self.generate_summary()

        # 保存报告
        if output_path:
            self.save_report(output_path)

        return True


def main():
    """主函数"""
    # 模型路径（根据执行计划）
    MODEL_PATH = "/mnt/hdd_4tb_data/ArchivedWorks/MBT/FTT/best_model.pth"
    OUTPUT_PATH = "/mnt/hdd_4tb_data/ArchivedWorks/MBT/Final/WE-FTT/feasibility_check/step1_report.txt"

    print("=" * 80)
    print("可行性验证 - Step 1: 模型快速验证")
    print("=" * 80)
    print(f"模型路径: {MODEL_PATH}")
    print(f"输出路径: {OUTPUT_PATH}")
    print()

    verifier = ModelVerifier(MODEL_PATH)
    success = verifier.run(OUTPUT_PATH)

    if success:
        print("\n✅ Step 1 验证完成！")
        print(f"📄 详细报告: {OUTPUT_PATH}")
    else:
        print("\n❌ Step 1 验证失败！")

    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
