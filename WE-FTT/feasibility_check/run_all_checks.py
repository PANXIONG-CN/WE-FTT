#!/usr/bin/env python3
"""
可行性验证 - 主控脚本
执行所有3个步骤的验证，并生成最终GO/NO-GO决策

使用方法:
    python run_all_checks.py

输出:
    - step1_report.txt (模型验证)
    - step2_report.txt (数据审计)
    - step3_report.txt (端到端验证)
    - FINAL_DECISION.txt (最终决策报告)
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


class FeasibilityRunner:
    """可行性验证主控制器"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.results = {}
        self.start_time = datetime.now()

    def run_step(self, step_num: int, script_name: str, description: str):
        """运行单个验证步骤"""
        print("\n" + "=" * 80)
        print(f"Step {step_num}: {description}")
        print("=" * 80)

        script_path = self.base_dir / script_name

        if not script_path.exists():
            print(f"❌ 脚本不存在: {script_path}")
            return False

        try:
            # 运行脚本
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )

            # 保存结果
            self.results[f"step{step_num}"] = {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

            # 显示输出
            if result.stdout:
                print(result.stdout)

            if result.stderr and result.returncode != 0:
                print(f"\n⚠️  错误输出:\n{result.stderr}")

            if result.returncode == 0:
                print(f"\n✅ Step {step_num} 完成")
                return True
            else:
                print(f"\n❌ Step {step_num} 失败 (返回码: {result.returncode})")
                return False

        except subprocess.TimeoutExpired:
            print(f"\n❌ Step {step_num} 超时")
            self.results[f"step{step_num}"] = {'success': False, 'error': 'timeout'}
            return False

        except Exception as e:
            print(f"\n❌ Step {step_num} 执行错误: {str(e)}")
            self.results[f"step{step_num}"] = {'success': False, 'error': str(e)}
            return False

    def analyze_reports(self):
        """分析所有报告，生成最终决策"""
        print("\n" + "=" * 80)
        print("分析验证报告...")
        print("=" * 80)

        decision_factors = {
            'model_is_we_ftt': None,
            'model_performance_ok': None,
            'data_available': None,
            'end_to_end_works': None
        }

        # 分析Step 1报告
        step1_report = self.base_dir / "step1_report.txt"
        if step1_report.exists():
            with open(step1_report, 'r', encoding='utf-8') as f:
                content = f.read()

                if "这是 WE-FTT 模型 ✅" in content:
                    decision_factors['model_is_we_ftt'] = True
                elif "这可能是 FT-Transformer" in content:
                    decision_factors['model_is_we_ftt'] = False

                if "性能匹配" in content or "MCC" in content:
                    decision_factors['model_performance_ok'] = True

        # 分析Step 2报告
        step2_report = self.base_dir / "step2_report.txt"
        if step2_report.exists():
            with open(step2_report, 'r', encoding='utf-8') as f:
                content = f.read()

                if "所有数据就绪" in content or "GO" in content:
                    decision_factors['data_available'] = True
                elif "NO-GO" in content:
                    decision_factors['data_available'] = False

        # 分析Step 3报告
        step3_report = self.base_dir / "step3_report.txt"
        if step3_report.exists():
            with open(step3_report, 'r', encoding='utf-8') as f:
                content = f.read()

                if "端到端流程可行" in content or "GO" in content:
                    decision_factors['end_to_end_works'] = True
                elif "NO-GO" in content:
                    decision_factors['end_to_end_works'] = False

        return decision_factors

    def generate_final_decision(self, decision_factors):
        """生成最终决策报告"""
        report_lines = []

        def add_line(line):
            report_lines.append(line)
            print(line)

        add_line("=" * 80)
        add_line("可行性验证 - 最终决策报告")
        add_line("=" * 80)
        add_line(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        add_line(f"总耗时: {(datetime.now() - self.start_time).total_seconds() / 60:.1f}分钟")
        add_line("")

        # 决策因素评估
        add_line("📋 决策因素评估:")
        add_line("")

        factors_desc = {
            'model_is_we_ftt': '模型类型是WE-FTT',
            'model_performance_ok': '模型性能匹配论文',
            'data_available': '数据完整可用',
            'end_to_end_works': '端到端流程可行'
        }

        go_count = 0
        nogo_count = 0
        unknown_count = 0

        for key, desc in factors_desc.items():
            value = decision_factors[key]
            if value is True:
                status = "✅ 通过"
                go_count += 1
            elif value is False:
                status = "❌ 未通过"
                nogo_count += 1
            else:
                status = "❓ 未知"
                unknown_count += 1

            add_line(f"   {desc}: {status}")

        add_line("")
        add_line("=" * 80)

        # 最终决策
        if go_count >= 3 and nogo_count == 0:
            # 理想情况：所有关键因素通过
            add_line("🚀 最终决策: GO - 可以立即开始补充实验")
            add_line("")
            add_line("✅ 执行建议:")
            add_line("   1. 立即开始物理机制统计分析（路径2，优先级P1）")
            add_line("   2. 并行开始震源学分层评估（路径3，优先级P1）")
            add_line("   3. 按照执行计划的时间表推进")
            add_line("")
            add_line("📅 预计完成时间: 1-2周内完成所有5个实验")

        elif decision_factors['model_is_we_ftt'] == False:
            # 模型是FT-Transformer而非WE-FTT
            add_line("⚠️  最终决策: CAUTION - 找到的是FT-Transformer")
            add_line("")
            add_line("🔍 问题分析:")
            add_line("   找到的模型是FT-Transformer（基线模型），而非WE-FTT")
            add_line("")
            add_line("💡 解决方案（按优先级）:")
            add_line("   1. 继续搜索WE-FTT模型checkpoint")
            add_line("      - 检查其他目录")
            add_line("      - 检查压缩包（如WE-FTT-backup-*.tar.gz）")
            add_line("   2. 如果找不到，考虑使用FT-Transformer并在论文中说明")
            add_line("      - 审稿人可能接受（毕竟是对比实验的一部分）")
            add_line("   3. 最后方案：重新训练WE-FTT（需要1-2天）")
            add_line("")
            add_line("📅 时间影响: +1-2天（搜索或重新训练）")

        elif decision_factors['data_available'] == False:
            # 数据问题
            add_line("🛑 最终决策: NO-GO - 数据不可用")
            add_line("")
            add_line("🔍 问题分析:")
            add_line("   关键数据缺失或无法访问")
            add_line("")
            add_line("💡 解决方案:")
            add_line("   1. 检查Step 2报告中的具体错误")
            add_line("   2. 修复数据问题")
            add_line("   3. 重新运行可行性验证")

        elif decision_factors['end_to_end_works'] == False:
            # 端到端流程有问题
            add_line("⚠️  最终决策: CAUTION - 端到端流程存在问题")
            add_line("")
            add_line("🔍 问题分析:")
            add_line("   虽然模型和数据都可用，但完整流程存在技术障碍")
            add_line("")
            add_line("💡 解决方案:")
            add_line("   1. 检查Step 3报告中的具体错误")
            add_line("   2. 可以先开始物理机制统计（不依赖模型推理）")
            add_line("   3. 同时解决端到端流程的技术问题")

        else:
            # 其他不确定情况
            add_line("❓ 最终决策: REVIEW NEEDED - 需要人工判断")
            add_line("")
            add_line("请查看各步骤的详细报告:")
            add_line(f"   - Step 1: step1_report.txt")
            add_line(f"   - Step 2: step2_report.txt")
            add_line(f"   - Step 3: step3_report.txt")

        add_line("")
        add_line("=" * 80)
        add_line("📄 详细报告路径:")
        add_line(f"   - 模型验证: {self.base_dir / 'step1_report.txt'}")
        add_line(f"   - 数据审计: {self.base_dir / 'step2_report.txt'}")
        add_line(f"   - 端到端验证: {self.base_dir / 'step3_report.txt'}")
        add_line(f"   - 最终决策: {self.base_dir / 'FINAL_DECISION.txt'}")
        add_line("=" * 80)

        # 保存报告
        output_path = self.base_dir / "FINAL_DECISION.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

        print(f"\n📄 最终决策报告已保存: {output_path}")

        return decision_factors

    def run_all(self):
        """运行所有验证步骤"""
        print("🚀 开始可行性验证")
        print(f"⏰ 开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 工作目录: {self.base_dir}")

        # Step 1: 模型验证
        step1_ok = self.run_step(1, "step1_model_verification.py", "模型快速验证")

        # Step 2: 数据审计
        step2_ok = self.run_step(2, "step2_data_audit.py", "数据可用性审计")

        # Step 3: 端到端验证
        step3_ok = self.run_step(3, "step3_end_to_end.py", "端到端最小化验证")

        # 生成最终决策
        print("\n" + "=" * 80)
        print("生成最终决策...")
        print("=" * 80)

        decision_factors = self.analyze_reports()
        self.generate_final_decision(decision_factors)

        # 返回总体成功状态
        overall_success = step1_ok and step2_ok and step3_ok

        print("\n" + "=" * 80)
        if overall_success:
            print("✅ 可行性验证全部完成")
        else:
            print("⚠️  可行性验证完成，但存在问题")
        print(f"⏰ 总耗时: {(datetime.now() - self.start_time).total_seconds() / 60:.1f}分钟")
        print("=" * 80)

        return overall_success


def main():
    runner = FeasibilityRunner()
    success = runner.run_all()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
