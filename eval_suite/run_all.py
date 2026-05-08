"""
一键启动脚本 (Run All)
=====================

功能：
1. 命令行参数支持 (--accuracy-only, --performance-only, --groups)
2. Rich 库实现进度条和实时统计
3. 自动保存结果到 eval_suite/results/
4. 运行结束后打印 ASCII 汇总表格

三组实验 (G1-G3):
    G1: BASELINE        - 完整Schema + 基础Prompt（无RAG、无KECA、无自愈）
    G2: KNOWLEDGE_RAG   - RAG检索Schema + KECA知识增强（无自愈）
    G3: FULL_SYSTEM     - RAG + 知识增强 + Agent自愈

使用方法:
    python -m eval_suite.run_all              # 运行全部评测 (G1-G3)
    python -m eval_suite.run_all --accuracy-only  # 仅准确率评测
    python -m eval_suite.run_all --groups G1 G2 G3  # 仅运行指定实验组
    python -m eval_suite.run_all --test-mode  # 快速测试模式
"""

import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# === 路径处理 ===
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入评测模块
from eval_suite.bench_accuracy import AccuracyBenchmark
from eval_suite.bench_performance import PerformanceBenchmark
from eval_suite.visualizer import Visualizer, ASCIITablePrinter

# 尝试导入 Rich 用于美化输出
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("💡 提示: 安装 rich 库可获得更好的可视化体验: pip install rich")


class EvalSuiteRunner:
    """
    评测套件运行器
    
    整合准确率评测和性能评测，提供一键运行接口。
    支持三组实验 (G1-G3) 的选择性运行。
    """
    
    # 默认运行的实验组
    DEFAULT_GROUPS = ["G1", "G2", "G3"]
    
    def __init__(self, database: str = "northwind", agent_backend: str = "legacy"):
        """初始化运行器
        
        Args:
            database: 数据库名称 ("northwind" 或 "adventureworks")
        """
        self.database = database.lower()
        self.agent_backend = agent_backend.lower()
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
    
    def print_banner(self):
        """打印启动横幅"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ██████╗ ███████╗███████╗██████╗ ██╗███╗   ██╗███████╗██╗ ██████╗ ██╗  ██╗████████╗║
║     ██╔══██╗██╔════╝██╔════╝██╔══██╗██║████╗  ██║██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝║
║     ██║  ██║█████╗  █████╗  ██████╔╝██║██╔██╗ ██║███████╗██║██║  ███╗███████║   ██║   ║
║     ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██║██║╚██╗██║╚════██║██║██║   ██║██╔══██║   ██║   ║
║     ██████╔╝███████╗███████╗██║     ██║██║ ╚████║███████║██║╚██████╔╝██║  ██║   ██║   ║
║     ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ║
║                                                                              ║
║                    🔬 自动化评测套件 v3.0                                      ║
║                    📊 三组对比实验 (G1-G3) + 性能评测                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        if RICH_AVAILABLE:
            self.console.print(Panel(
                "[bold cyan]DeepInsight 自动化评测套件 v3.0[/bold cyan]\n"
                "[dim]G1(Baseline) / G2(Knowledge-RAG) / G3(Full) 三组对比实验[/dim]",
                title="🔬 Evaluation Suite",
                border_style="blue"
            ))
        else:
            print(banner)
    
    def print_phase(self, phase: str, description: str):
        """打印阶段信息"""
        if RICH_AVAILABLE:
            self.console.rule(f"[bold blue]{phase}[/bold blue]")
            self.console.print(f"[dim]{description}[/dim]\n")
        else:
            print(f"\n{'='*60}")
            print(f"  {phase}")
            print(f"  {description}")
            print('='*60 + "\n")
    
    def run_accuracy_benchmark(
        self, 
        limit: Optional[int] = None,
        runs: int = 1,
        groups: Optional[List[str]] = None,
        verbose: bool = True
    ) -> dict:
        """
        运行准确率评测 (G1-G3 三组实验)
        
        Args:
            limit: 每组实验的用例数量限制
            runs: 运行次数（多次运行取平均，每轮创建新 Agent）
            groups: 要运行的实验组列表 (默认 ["G1", "G2", "G3"])
            verbose: 详细输出
            
        Returns:
            三组实验的摘要字典
        """
        groups = groups or self.DEFAULT_GROUPS
        group_names = " / ".join(groups)
        
        self.print_phase("Phase 1/3: Accuracy Benchmark", 
                        f"评测 {group_names} 实验组的准确率 (运行 {runs} 轮)")
        
        all_runs_results = []
        
        for run_idx in range(runs):
            if runs > 1 and verbose:
                print(f"\n--- 第 {run_idx + 1}/{runs} 轮运行 ---")
            
            # 每轮创建新的 AccuracyBenchmark（即每轮创建新 Agent）
            benchmark = AccuracyBenchmark(database=self.database, agent_backend=self.agent_backend)
            
            # 运行前验证 Agent
            if run_idx == 0:
                verification = benchmark.verify_all_agents()
                all_ready = all(ok for ok, _ in verification.values())
                if not all_ready:
                    print("⚠️ 警告：部分 Agent 创建失败，将使用模拟模式")
            
            if RICH_AVAILABLE and verbose and runs == 1:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    console=self.console
                ) as progress:
                    task = progress.add_task("[cyan]运行准确率评测...", total=len(groups))
                    
                    summaries = {}
                    g2_results = None
                    
                    for group_id in groups:
                        progress.update(task, description=f"[cyan]{group_id} 实验...")
                        
                        if group_id.upper() == "G3" and g2_results is not None:
                            # P4: G3 使用共享首次生成
                            summaries[group_id.lower()] = benchmark.run_full_system_with_g2_results(
                                g2_results, verbose=False
                            )
                        else:
                            summaries[group_id.lower()] = benchmark.run_experiment(group_id, limit, verbose=False)
                        
                        # 保存 G2 结果供 G3 使用
                        if group_id.upper() == "G2":
                            from eval_suite.bench_accuracy import ExperimentMode
                            g2_results = benchmark.results.get(ExperimentMode.KNOWLEDGE_RAG, [])
                        
                        progress.advance(task)
            else:
                summaries = {}
                g2_results = None
                
                for group_id in groups:
                    if group_id.upper() == "G3" and g2_results is not None:
                        # P4: G3 使用共享首次生成
                        summaries[group_id.lower()] = benchmark.run_full_system_with_g2_results(
                            g2_results, verbose=False
                        )
                    else:
                        summaries[group_id.lower()] = benchmark.run_experiment(group_id, limit, verbose=False)
                    
                    # 保存 G2 结果供 G3 使用
                    if group_id.upper() == "G2":
                        from eval_suite.bench_accuracy import ExperimentMode
                        g2_results = benchmark.results.get(ExperimentMode.KNOWLEDGE_RAG, [])
            
            all_runs_results.append(summaries)
        
        # 计算多轮平均
        if runs > 1:
            averaged_results = self._average_accuracy_results(all_runs_results, groups)
            # 保存平均结果到 JSON 文件，供 visualizer 使用
            self._save_averaged_results(averaged_results)
            return averaged_results
        else:
            return all_runs_results[0]
    
    def _save_averaged_results(self, averaged_results: dict):
        """
        保存平均结果到 JSON 文件，供 visualizer 使用
        
        Args:
            averaged_results: 平均后的结果字典
        """
        import json
        from dataclasses import asdict
        
        results_dict = {}
        for mode, summary in averaged_results.items():
            results_dict[mode] = asdict(summary)
        
        json_path = self.results_dir / "averaged_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        
        print(f"💾 平均结果已保存到: {json_path}")
    
    def _average_accuracy_results(self, all_runs: list, groups: List[str]) -> dict:
        """
        计算多轮运行的平均准确率和标准差
        
        Args:
            all_runs: 每轮运行结果列表
            groups: 运行的实验组列表
            
        Returns:
            平均后的结果字典
        """
        import math
        
        averaged = {}
        
        print("\n" + "=" * 60)
        print(f"📊 多轮实验平均结果 ({len(all_runs)} 轮)")
        print("=" * 60)
        
        for group_id in groups:
            mode = group_id.lower()
            if mode not in all_runs[0]:
                continue
            
            # 计算总体准确率的平均和标准差
            accuracies = [run[mode].accuracy for run in all_runs]
            avg_accuracy = sum(accuracies) / len(accuracies)
            
            if len(accuracies) > 1:
                variance = sum((a - avg_accuracy) ** 2 for a in accuracies) / (len(accuracies) - 1)
                std_accuracy = math.sqrt(variance)
            else:
                std_accuracy = 0.0
            
            # 计算各难度级别的平均准确率
            avg_by_difficulty = {}
            difficulties = ['easy', 'medium', 'hard', 'extra_hard']
            
            for diff in difficulties:
                diff_accuracies = []
                for run in all_runs:
                    if diff in run[mode].by_difficulty:
                        diff_accuracies.append(run[mode].by_difficulty[diff]['accuracy'])
                
                if diff_accuracies:
                    avg_diff_acc = sum(diff_accuracies) / len(diff_accuracies)
                    avg_by_difficulty[diff] = {
                        'total': all_runs[-1][mode].by_difficulty.get(diff, {}).get('total', 0),
                        'correct': int(avg_diff_acc * all_runs[-1][mode].by_difficulty.get(diff, {}).get('total', 0)),
                        'accuracy': avg_diff_acc
                    }
            
            # 计算 Token 消耗的平均值
            avg_tokens_per_query = sum(run[mode].avg_tokens_per_query for run in all_runs) / len(all_runs)

            # 计算响应时间的平均值（含 P95）
            avg_exec_time = sum(run[mode].avg_execution_time_ms for run in all_runs) / len(all_runs)
            avg_p95_time = sum(run[mode].p95_execution_time_ms for run in all_runs) / len(all_runs)

            # 创建平均结果
            result = all_runs[-1][mode]
            result.accuracy = avg_accuracy
            result.accuracy_std = std_accuracy
            result.run_count = len(all_runs)
            result.by_difficulty = avg_by_difficulty
            result.avg_tokens_per_query = avg_tokens_per_query
            result.avg_execution_time_ms = avg_exec_time
            result.p95_execution_time_ms = avg_p95_time
            
            averaged[mode] = result
            
            print(f"\n{group_id}:")
            print(f"   总体准确率: {avg_accuracy*100:.1f}% ± {std_accuracy*100:.1f}%")
            for diff in difficulties:
                if diff in avg_by_difficulty:
                    print(f"   {diff:12s}: {avg_by_difficulty[diff]['accuracy']*100:.1f}%")
            print(f"   平均 Tokens: {avg_tokens_per_query:.0f}")
            print(f"   平均响应时间: {avg_exec_time:.1f} ms  |  P95: {avg_p95_time:.1f} ms")
        
        print("=" * 60)
        return averaged
    
    def run_performance_benchmark(
        self, 
        samples: int = 3,
        runs: int = 1,
        verbose: bool = True
    ) -> dict:
        """
        运行性能评测（支持多轮运行取平均）
        
        Args:
            samples: 每个查询的采样次数
            runs: 运行次数（多次运行取平均）
            verbose: 详细输出
            
        Returns:
            各后端的性能摘要
        """
        self.print_phase("Phase 2/3: Performance Benchmark",
                        f"对比 PyTorch FP32 / OpenVINO FP32 Embedding 推理性能 (运行 {runs} 轮)")
        
        all_runs_summaries = []
        
        for run_idx in range(runs):
            if runs > 1 and verbose:
                print(f"\n--- 性能评测第 {run_idx + 1}/{runs} 轮运行 ---")
            
            benchmark = PerformanceBenchmark()
            
            # 直接使用 bench_performance 的 run_all 方法，避免重复逻辑
            summaries = benchmark.run_all(samples_per_query=samples, verbose=verbose)
            all_runs_summaries.append(summaries)
        
        # 计算多轮平均（如果 runs > 1）
        if runs > 1:
            averaged_summaries = self._average_performance_results(all_runs_summaries)
            # 保存平均结果
            self._save_averaged_performance_results(averaged_summaries)
            return averaged_summaries
        else:
            return all_runs_summaries[0]
    
    def _average_performance_results(self, all_runs: list) -> dict:
        """
        计算性能评测的多轮运行平均结果
        
        Args:
            all_runs: 每轮运行结果列表
            
        Returns:
            平均后的性能摘要字典
        """
        import math
        from dataclasses import asdict
        
        if not all_runs:
            return {}
        
        backends = list(all_runs[0].keys())
        averaged = {}
        
        print("\n" + "=" * 60)
        print(f"⚡ 性能评测多轮平均结果 ({len(all_runs)} 轮)")
        print("=" * 60)
        
        for backend in backends:
            # 收集所有轮次的该后端数据
            all_latencies = []
            all_p50s = []
            all_p95s = []
            all_throughputs = []
            all_memories = []
            
            for run in all_runs:
                if backend in run:
                    s = run[backend]
                    all_latencies.append(s.latency_avg_ms)
                    all_p50s.append(s.latency_p50_ms)
                    all_p95s.append(s.latency_p95_ms)
                    all_throughputs.append(s.throughput_qps)
            
            if not all_latencies:
                continue
            
            # 计算平均值
            avg_latency = sum(all_latencies) / len(all_latencies)
            avg_p50 = sum(all_p50s) / len(all_p50s)
            avg_p95 = sum(all_p95s) / len(all_p95s)
            avg_throughput = sum(all_throughputs) / len(all_throughputs)
            
            # 计算标准差
            def calc_std(values, avg):
                if len(values) <= 1:
                    return 0.0
                variance = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
                return math.sqrt(variance)
            
            std_latency = calc_std(all_latencies, avg_latency)
            
            # 从最后一轮复制对象结构并更新值
            result = all_runs[-1][backend]
            result.latency_avg_ms = avg_latency
            result.latency_p50_ms = avg_p50
            result.latency_p95_ms = avg_p95
            result.throughput_qps = avg_throughput
            result.run_count = len(all_runs)
            result.latency_std_ms = std_latency
            
            averaged[backend] = result
            
            print(f"\n{backend}:")
            print(f"   平均延迟: {avg_latency:.2f}ms ± {std_latency:.2f}ms")
            print(f"   平均 P50: {avg_p50:.2f}ms")
            print(f"   平均 P95: {avg_p95:.2f}ms")
            print(f"   平均吞吐量: {avg_throughput:.1f} QPS")
        
        # 重新计算加速比
        if 'pytorch_fp32' in averaged:
            baseline_lat = averaged['pytorch_fp32'].latency_avg_ms
            for name, summary in averaged.items():
                if summary.latency_avg_ms > 0:
                    summary.speedup_vs_baseline = baseline_lat / summary.latency_avg_ms
        
        print("=" * 60)
        return averaged
    
    def _save_averaged_performance_results(self, averaged_results: dict):
        """
        保存性能评测的平均结果
        
        Args:
            averaged_results: 平均后的结果字典
        """
        import json
        from dataclasses import asdict
        
        results_dict = {}
        for backend, summary in averaged_results.items():
            results_dict[backend] = asdict(summary)
        
        json_path = self.results_dir / "averaged_performance_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        
        print(f"💾 性能评测平均结果已保存到: {json_path}")
    
    def generate_reports(self, verbose: bool = True):
        """
        生成可视化报告
        
        Args:
            verbose: 详细输出
        """
        self.print_phase("Phase 3/3: Report Generation",
                        "生成 Markdown 报告、Plotly 图表、ASCII 表格")
        
        visualizer = Visualizer()
        visualizer.generate_all(verbose)
    
    def print_summary(
        self, 
        accuracy_results: dict = None, 
        performance_results: dict = None
    ):
        """打印最终汇总"""
        if RICH_AVAILABLE:
            self.console.print("\n")
            self.console.rule("[bold green]评测完成[/bold green]")
            
            # 准确率表格
            if accuracy_results:
                # 检查是否有多轮运行
                any_multi_run = any(getattr(summary, 'run_count', 1) > 1 for summary in accuracy_results.values())
                title = "📊 准确率评测结果 (EX)"
                if any_multi_run:
                    run_count = max(getattr(summary, 'run_count', 1) for summary in accuracy_results.values())
                    title = f"📊 准确率评测结果 (EX) - {run_count} 轮平均"
                table = Table(title=title, box=box.ROUNDED)
                table.add_column("方案", style="cyan")
                table.add_column("Easy", justify="center")
                table.add_column("Medium", justify="center")
                table.add_column("Hard", justify="center")
                table.add_column("Extra-Hard", justify="center")
                table.add_column("Overall", justify="center", style="bold green")
                table.add_column("Avg Tokens", justify="center", style="dim")  # 🆕 Token 列
                
                for mode, summary in accuracy_results.items():
                    row = [mode]
                    for diff in ['easy', 'medium', 'hard', 'extra_hard']:
                        if diff in summary.by_difficulty:
                            row.append(f"{summary.by_difficulty[diff]['accuracy']*100:.1f}%")
                        else:
                            row.append("-")
                    # 显示总体准确率，如果有多轮运行则显示标准差
                    run_count = getattr(summary, 'run_count', 1)
                    if run_count > 1:
                        std_acc = getattr(summary, 'accuracy_std', 0)
                        row.append(f"{summary.accuracy*100:.1f}%±{std_acc*100:.1f}%")
                    else:
                        row.append(f"{summary.accuracy*100:.1f}%")
                    # 🆕 添加平均 Token 消耗
                    avg_tokens = getattr(summary, 'avg_tokens_per_query', 0)
                    row.append(f"{avg_tokens:.0f}" if avg_tokens > 0 else "-")
                    table.add_row(*row)
                
                self.console.print(table)
            
            # 性能表格
            if performance_results:
                # 检查是否有多轮运行
                any_multi_run = any(getattr(summary, 'run_count', 1) > 1 for summary in performance_results.values())
                title = "⚡ 性能评测结果"
                if any_multi_run:
                    run_count = max(getattr(summary, 'run_count', 1) for summary in performance_results.values())
                    title = f"⚡ 性能评测结果 - {run_count} 轮平均"
                table = Table(title=title, box=box.ROUNDED)
                table.add_column("后端", style="cyan")
                table.add_column("延迟(ms)", justify="center")
                table.add_column("P95(ms)", justify="center")
                table.add_column("QPS", justify="center")
                table.add_column("加速比", justify="center", style="bold green")
                
                for backend, summary in performance_results.items():
                    # 显示延迟，如果有多轮运行则显示标准差
                    run_count = getattr(summary, 'run_count', 1)
                    if run_count > 1:
                        std_lat = getattr(summary, 'latency_std_ms', 0)
                        latency_str = f"{summary.latency_avg_ms:.1f}±{std_lat:.1f}"
                    else:
                        latency_str = f"{summary.latency_avg_ms:.1f}"
                    table.add_row(
                        backend,
                        latency_str,
                        f"{summary.latency_p95_ms:.1f}",
                        f"{summary.throughput_qps:.1f}",
                        f"{summary.speedup_vs_baseline:.2f}x"
                    )
                
                self.console.print(table)
            
            # 输出路径
            self.console.print(f"\n💾 结果已保存到: [bold]{self.results_dir}[/bold]")
            self.console.print("   - 📄 evaluation_report.md (Markdown 报告)")
            self.console.print("   - 📊 charts/ (Plotly 交互图表)")
            self.console.print("   - 📁 *.csv (详细数据)")
        else:
            # 使用 ASCII 表格
            printer = ASCIITablePrinter()
            
            if accuracy_results:
                # 转换格式
                acc_data = {}
                for mode, summary in accuracy_results.items():
                    acc_data[mode] = {
                        'accuracy': summary.accuracy,
                        'by_difficulty': summary.by_difficulty
                    }
                print(printer.accuracy_table(acc_data))
            
            if performance_results:
                # 转换格式
                perf_data = {}
                for backend, summary in performance_results.items():
                    perf_data[backend] = {
                        'latency_avg_ms': summary.latency_avg_ms,
                        'latency_p95_ms': summary.latency_p95_ms,
                        'throughput_qps': summary.throughput_qps,
                        'speedup_vs_baseline': summary.speedup_vs_baseline
                    }
                print(printer.performance_table(perf_data))
            
            print(f"\n💾 结果已保存到: {self.results_dir}")
    
    def run(
        self,
        accuracy_only: bool = False,
        performance_only: bool = False,
        test_mode: bool = False,
        limit: Optional[int] = None,
        runs: int = 1,
        samples: int = 3,
        groups: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """
        运行完整评测流程
        
        Args:
            accuracy_only: 仅运行准确率评测
            performance_only: 仅运行性能评测
            test_mode: 测试模式（快速运行）
            limit: 准确率评测用例限制
            runs: 准确率评测运行次数（多次运行取平均）
            samples: 性能评测采样次数
            groups: 要运行的实验组列表 (默认 ["G1", "G2", "G3"])
            verbose: 详细输出
        """
        start_time = time.time()
        
        self.print_banner()
        
        groups = groups or self.DEFAULT_GROUPS
        print(f"\n⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📋 实验组: {' / '.join(groups)}")
        print(f"🧠 Agent Backend: {self.agent_backend}")
        
        if test_mode:
            limit = limit or 3
            runs = 1  # 测试模式仅运行1轮
            samples = 1
            print("🧪 测试模式: 使用最小样本量快速验证")
        
        accuracy_results = None
        performance_results = None
        
        # 准确率评测
        if not performance_only:
            accuracy_results = self.run_accuracy_benchmark(limit, runs, groups, verbose)
        
        # 性能评测
        if not accuracy_only:
            performance_results = self.run_performance_benchmark(samples, runs, verbose)
        
        # 生成报告
        self.generate_reports(verbose)
        
        # 打印汇总
        self.print_summary(accuracy_results, performance_results)
        
        elapsed = time.time() - start_time
        print(f"\n⏱️ 总耗时: {elapsed:.1f} 秒")
        print("\n✅ 评测完成！")


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="DeepInsight 自动化评测套件 (G1-G3 三组实验)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python -m eval_suite.run_all              # 运行全部评测 (G1-G3)
    python -m eval_suite.run_all --accuracy-only  # 仅准确率评测
    python -m eval_suite.run_all --groups G1 G2 G3  # 仅运行指定实验组
    python -m eval_suite.run_all --performance-only  # 仅性能评测
    python -m eval_suite.run_all --test-mode  # 快速测试模式
    python -m eval_suite.run_all --limit 10   # 限制评测题目数量
        """
    )
    
    parser.add_argument("--accuracy-only", action="store_true",
                        help="仅运行准确率评测")
    parser.add_argument("--performance-only", action="store_true",
                        help="仅运行性能评测")
    parser.add_argument("--test-mode", action="store_true",
                        help="测试模式（最小样本量快速验证）")
    parser.add_argument("--limit", type=int, default=None,
                        help="准确率评测的题目数量限制")
    parser.add_argument("--runs", type=int, default=1,
                        help="准确率评测运行次数（多次运行取平均）")
    parser.add_argument("--samples", type=int, default=3,
                        help="性能评测每查询采样次数")
    parser.add_argument("--groups", nargs="+", default=None,
                        choices=["G1", "G2", "G3"],
                        help="要运行的实验组 (默认 G1 G2 G3)")
    parser.add_argument("--database", type=str, default="northwind",
                        choices=["northwind", "adventureworks"],
                        help="使用的数据库 (northwind 或 adventureworks)")
    parser.add_argument("--quiet", action="store_true",
                        help="安静模式，减少输出")
    parser.add_argument("--agent-backend", choices=["legacy", "graph"], default="legacy",
                        help="Agent 后端：legacy 使用原 Text2SQLAgent，graph 使用实验性 LangGraph 路径")
    
    args = parser.parse_args()
    
    runner = EvalSuiteRunner(database=args.database, agent_backend=args.agent_backend)
    runner.run(
        accuracy_only=args.accuracy_only,
        performance_only=args.performance_only,
        test_mode=args.test_mode,
        limit=args.limit,
        runs=args.runs,
        samples=args.samples,
        groups=args.groups,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
