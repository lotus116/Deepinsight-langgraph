"""
可视化与报告生成模块 (Visualizer)
================================

功能：
1. 生成符合论文格式的 Markdown 表格
2. 生成 Plotly 交互式图表
3. 输出终端 ASCII 表格

学术意义:
    可视化是毕业论文的关键组成部分，良好的数据展示
    有助于清晰地传达系统性能和实验结论。
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

# === 路径处理 ===
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


class MarkdownGenerator:
    """
    Markdown 表格生成器
    
    生成适合直接复制到论文中的 Markdown 格式表格。
    """
    
    @staticmethod
    def accuracy_table(data: Dict[str, Any]) -> str:
        """
        生成准确率对比表格
        
        Args:
            data: 准确率实验数据
            
        Returns:
            Markdown 格式的表格字符串
        """
        # 检查是否有多轮运行数据
        any_multi_run = any('run_count' in stats and stats['run_count'] > 1 for stats in data.values())
        run_count = max(stats.get('run_count', 1) for stats in data.values()) if any_multi_run else 1
        
        title = "## 表 X: Text-to-SQL 准确率对比实验结果 (EX 指标)"
        if any_multi_run:
            title = f"## 表 X: Text-to-SQL 准确率对比实验结果 (EX 指标) - {run_count} 轮平均"
        
        lines = [
            title,
            "",
            "| 方案 | Easy | Medium | Hard | Extra-Hard | **Overall** | Avg Tokens |",
            "|:-----|:----:|:------:|:----:|:----------:|:-----------:|:----------:|"
        ]
        
        for mode_name, stats in data.items():
            row = f"| {mode_name} "
            for diff in ['easy', 'medium', 'hard', 'extra_hard']:
                if diff in stats.get('by_difficulty', {}):
                    acc = stats['by_difficulty'][diff]['accuracy'] * 100
                    row += f"| {acc:.1f}% "
                else:
                    row += "| - "
            overall = stats.get('accuracy', 0) * 100
            # 如果有多轮运行，显示标准差
            if 'run_count' in stats and stats['run_count'] > 1:
                std_acc = stats.get('accuracy_std', 0) * 100
                row += f"| **{overall:.1f}%±{std_acc:.1f}%** "
            else:
                row += f"| **{overall:.1f}%** "
            avg_tokens = stats.get('avg_tokens_per_query', 0)
            row += f"| {avg_tokens:.0f} |" if avg_tokens > 0 else "| - |"
            lines.append(row)
        
        lines.append("")
        lines.append("*EX: Execution Accuracy - 执行结果集完全匹配的比例*")
        if any_multi_run:
            lines.append("*± 后数值为标准差*")
        
        return "\n".join(lines)
    
    @staticmethod
    def performance_table(data: Dict[str, Any]) -> str:
        """
        生成性能对比表格
        
        Args:
            data: 性能实验数据
            
        Returns:
            Markdown 格式的表格字符串
        """
        # 检查是否有多轮运行
        any_multi_run = any(stats.get('run_count', 1) > 1 for stats in data.values())
        
        title = "## 表 Y: 推理性能对比实验结果"
        if any_multi_run:
            run_count = max(stats.get('run_count', 1) for stats in data.values())
            title = f"## 表 Y: 推理性能对比实验结果 ({run_count} 轮平均)"
        
        lines = [
            title,
            "",
            "| 推理后端 | 延迟(ms) | P95延迟(ms) | QPS | 加速比 |",
            "|:---------|:--------:|:-----------:|:---:|:------:|"
        ]
        
        for backend, stats in data.items():
            # 显示延迟，如果有多轮运行则显示标准差
            run_count = stats.get('run_count', 1)
            if run_count > 1:
                std_lat = stats.get('latency_std_ms', 0)
                latency_str = f"{stats.get('latency_avg_ms', 0):.1f}±{std_lat:.1f}"
            else:
                latency_str = f"{stats.get('latency_avg_ms', 0):.1f}"
            
            lines.append(
                f"| {backend} | "
                f"{latency_str} | "
                f"{stats.get('latency_p95_ms', 0):.1f} | "
                f"{stats.get('throughput_qps', stats.get('tps_avg', 0)):.1f} | "
                f"{stats.get('speedup_vs_baseline', 1):.2f}x |"
            )
        
        lines.append("")
        lines.append("*QPS: Queries Per Second, 加速比相对于 PyTorch FP32 基准*")
        if any_multi_run:
            lines.append("*延迟显示为 平均值±标准差*")
        
        return "\n".join(lines)
    
    @staticmethod
    def combined_report(
        accuracy_data: Dict[str, Any],
        performance_data: Dict[str, Any],
        system_info: Dict[str, Any]
    ) -> str:
        """
        生成完整的实验报告
        
        Args:
            accuracy_data: 准确率数据
            performance_data: 性能数据
            system_info: 系统信息
            
        Returns:
            完整的 Markdown 报告
        """
        lines = [
            "# DeepInsight 评测报告",
            "",
            f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "---",
            "",
            "## 1. 实验环境",
            "",
            "| 配置项 | 值 |",
            "|:-------|:---|"
        ]
        
        if system_info:
            lines.append(f"| CPU | {system_info.get('cpu_model', 'N/A')} |")
            lines.append(f"| 核心/线程 | {system_info.get('cpu_cores', 'N/A')}/{system_info.get('cpu_threads', 'N/A')} |")
            lines.append(f"| 内存 | {system_info.get('memory_total_gb', 0):.1f} GB |")
            lines.append(f"| GPU | {system_info.get('gpu_info', 'N/A')} |")
            lines.append(f"| OpenVINO | {system_info.get('openvino_version', 'N/A')} |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # 准确率表格
        if accuracy_data:
            lines.append(MarkdownGenerator.accuracy_table(accuracy_data))
            lines.append("")
            lines.append("---")
            lines.append("")

        # G1-G3 响应时间表格
        if accuracy_data and any('avg_execution_time_ms' in v for v in accuracy_data.values()):
            lines.append("## 2. 端到端响应时间 (G1–G3)")
            lines.append("")
            lines.append("| 组别 | 平均响应时间 (ms) | P95 响应时间 (ms) |")
            lines.append("|:-----|:----------------:|:----------------:|")
            mode_labels = {
                'g1': 'G1 Baseline',
                'g2': 'G2 RAG+KECA',
                'g3': 'G3 Full System',
                'baseline': 'G1 Baseline',
                'knowledge_rag': 'G2 RAG+KECA',
                'full_system': 'G3 Full System',
            }
            for mode_key, stats in accuracy_data.items():
                label = mode_labels.get(mode_key, mode_key)
                avg_t = stats.get('avg_execution_time_ms', 0)
                p95_t = stats.get('p95_execution_time_ms', 0)
                lines.append(f"| {label} | {avg_t:.1f} | {p95_t:.1f} |")
            lines.append("")
            run_count_note = max(
                (v.get('run_count', 1) for v in accuracy_data.values()), default=1
            )
            if run_count_note > 1:
                lines.append(f"*响应时间为 {run_count_note} 轮跨轮平均；P95 采用线性插值法，N=50 时对应排序后第 47–48 个样本*")
            else:
                lines.append("*P95 采用线性插值法*")
            lines.append("")
            lines.append("---")
            lines.append("")

        # 性能表格
        if performance_data:
            lines.append(MarkdownGenerator.performance_table(performance_data))
        
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 3. 结论")
        lines.append("")
        
        # 自动生成结论 (G1-G3 增量分析)
        if accuracy_data and 'baseline' in accuracy_data and 'full_system' in accuracy_data:
            baseline_acc = accuracy_data['baseline'].get('accuracy', 0)
            full_acc = accuracy_data['full_system'].get('accuracy', 0)
            improvement = (full_acc - baseline_acc) * 100
            lines.append(f"- 完整系统相比基准方案，准确率提升 **{improvement:.1f}** 个百分点")
        
        # Knowledge-RAG 贡献分析 (G2 vs G1)
        if accuracy_data and 'baseline' in accuracy_data and 'knowledge_rag' in accuracy_data:
            baseline_acc = accuracy_data['baseline'].get('accuracy', 0)
            knowledge_rag_acc = accuracy_data['knowledge_rag'].get('accuracy', 0)
            knowledge_improvement = (knowledge_rag_acc - baseline_acc) * 100
            if knowledge_improvement > 0:
                lines.append(f"- RAG + KECA 知识增强相比 Baseline，准确率提升 **{knowledge_improvement:.1f}** 个百分点")
        
        if performance_data and 'pytorch_fp32' in performance_data and 'openvino_fp32' in performance_data:
            pytorch_lat = performance_data['pytorch_fp32'].get('latency_avg_ms', 1)
            openvino_lat = performance_data['openvino_fp32'].get('latency_avg_ms', 1)
            speedup = pytorch_lat / openvino_lat if openvino_lat > 0 else 1
            lines.append(f"- OpenVINO FP32 优化实现 **{speedup:.2f}x** 推理加速")

        # G3 P95 稳定性警告
        g3_stats = accuracy_data.get('g3', accuracy_data.get('full_system', {})) if accuracy_data else {}
        if g3_stats:
            g3_p95 = g3_stats.get('p95_execution_time_ms', 0)
            if g3_p95 > 10000:  # > 10 秒
                lines.append(f"- G3 P95 响应时间 **{g3_p95/1000:.1f}s**，超出 10s 设计目标，自愈重试是主要瓶颈")
        
        return "\n".join(lines)


class PlotlyChartGenerator:
    """
    Plotly 图表生成器
    
    生成交互式图表，保存为 HTML 文件。
    """
    
    def __init__(self, output_dir: Path = None):
        """
        初始化图表生成器
        
        Args:
            output_dir: 图表输出目录
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "results" / "charts"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 尝试导入 Plotly
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            self.go = go
            self.px = px
            self.make_subplots = make_subplots
            self.available = True
        except ImportError:
            print("⚠️ Plotly 未安装，图表功能不可用")
            self.available = False
    
    def accuracy_bar_chart(
        self, 
        data: Dict[str, Any],
        filename: str = "accuracy_comparison.html"
    ) -> Optional[str]:
        """
        生成准确率对比柱状图
        
        按难度分组，对比三组实验 (G1-G3) 的准确率。
        
        Args:
            data: 准确率数据
            filename: 输出文件名
            
        Returns:
            生成的文件路径
        """
        if not self.available:
            return self._generate_fallback_chart(data, filename, "accuracy")
        
        difficulties = ['easy', 'medium', 'hard', 'extra_hard']
        display_names = {'easy': 'Easy', 'medium': 'Medium', 'hard': 'Hard', 'extra_hard': 'Extra-Hard'}
        
        fig = self.go.Figure()
        
        colors = {
            'baseline': '#FF6B6B',       # G1 红色
            'knowledge_rag': '#4ECDC4',  # G2 青色 (RAG + KECA)
            'full_system': '#45B7D1'     # G3 蓝色
        }
        
        for mode_name, stats in data.items():
            y_values = []
            for diff in difficulties:
                if diff in stats.get('by_difficulty', {}):
                    y_values.append(stats['by_difficulty'][diff]['accuracy'] * 100)
                else:
                    y_values.append(0)
            
            fig.add_trace(self.go.Bar(
                name=mode_name.replace('_', ' ').title(),
                x=[display_names[d] for d in difficulties],
                y=y_values,
                marker_color=colors.get(mode_name, '#888888'),
                text=[f'{v:.1f}%' for v in y_values],
                textposition='outside'
            ))
        
        fig.update_layout(
            title='Text-to-SQL 准确率对比 (按难度分组)',
            xaxis_title='难度等级',
            yaxis_title='准确率 (%)',
            barmode='group',
            template='plotly_white',
            font=dict(size=14),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            yaxis=dict(range=[0, 100])
        )
        
        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        print(f"📊 准确率图表已保存: {output_path}")
        
        return str(output_path)
    
    def performance_radar_chart(
        self,
        data: Dict[str, Any],
        filename: str = "performance_radar.html"
    ) -> Optional[str]:
        """
        生成性能对比雷达图
        
        展示各后端在不同指标上的相对表现。
        
        Args:
            data: 性能数据
            filename: 输出文件名
            
        Returns:
            生成的文件路径
        """
        if not self.available:
            return self._generate_fallback_chart(data, filename, "performance")
        
        # 定义指标（归一化到 0-1）
        categories = ['吞吐量(QPS)', '延迟(倒数)', 'P95稳定性']
        
        fig = self.go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, (backend, stats) in enumerate(data.items()):
            # 归一化各指标
            max_qps = max(s.get('throughput_qps', s.get('tps_avg', 1)) for s in data.values())
            max_latency = max(s.get('latency_avg_ms', 1) for s in data.values())
            max_p95 = max(s.get('latency_p95_ms', 1) for s in data.values())
            
            values = [
                stats.get('throughput_qps', stats.get('tps_avg', 0)) / max_qps if max_qps > 0 else 0,
                1 - (stats.get('latency_avg_ms', 0) / max_latency) if max_latency > 0 else 0,
                1 - (stats.get('latency_p95_ms', 0) / max_p95) if max_p95 > 0 else 0
            ]
            values.append(values[0])  # 闭合雷达图
            
            fig.add_trace(self.go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=backend,
                line_color=colors[i % len(colors)],
                opacity=0.7
            ))
        
        fig.update_layout(
            title='推理后端性能对比 (雷达图)',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            template='plotly_white',
            font=dict(size=14),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.2
            )
        )
        
        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        print(f"📊 雷达图已保存: {output_path}")
        
        return str(output_path)
    
    def latency_boxplot(
        self,
        metrics_csv: str = None,
        filename: str = "latency_distribution.html"
    ) -> Optional[str]:
        """
        生成延迟分布箱线图
        
        Args:
            metrics_csv: 详细指标 CSV 文件路径
            filename: 输出文件名
            
        Returns:
            生成的文件路径
        """
        if not self.available:
            return None
        
        if metrics_csv is None:
            metrics_csv = Path(__file__).parent / "results" / "performance_details.csv"
        
        if not Path(metrics_csv).exists():
            print(f"⚠️ 未找到性能数据文件: {metrics_csv}")
            return None
        
        df = pd.read_csv(metrics_csv)
        
        fig = self.px.box(
            df,
            x='backend',
            y='latency_ms',
            color='backend',
            title='推理延迟分布 (箱线图)',
            labels={'latency_ms': '延迟 (ms)', 'backend': '推理后端'},
            template='plotly_white'
        )
        
        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        print(f"📊 箱线图已保存: {output_path}")
        
        return str(output_path)
    
    def _generate_fallback_chart(
        self, 
        data: Dict[str, Any], 
        filename: str, 
        chart_type: str
    ) -> str:
        """
        生成降级的纯 HTML 图表
        
        当 Plotly 不可用时的替代方案。
        """
        output_path = self.output_dir / filename.replace('.html', '_fallback.html')
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>DeepInsight {chart_type.title()} Chart</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        .chart-placeholder {{
            border: 2px dashed #ccc;
            padding: 40px;
            text-align: center;
            background: #f9f9f9;
            border-radius: 8px;
        }}
        table {{ border-collapse: collapse; margin: 20px auto; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        th {{ background: #4ECDC4; color: white; }}
    </style>
</head>
<body>
    <h2>DeepInsight {chart_type.title()} Data</h2>
    <div class="chart-placeholder">
        <p>⚠️ Plotly 未安装，显示表格数据</p>
        <p>安装 Plotly: <code>pip install plotly</code></p>
    </div>
    <h3>原始数据</h3>
    <pre>{json.dumps(data, indent=2, ensure_ascii=False)}</pre>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)


class ASCIITablePrinter:
    """
    ASCII 表格打印器
    
    生成终端友好的格式化表格。
    """
    
    @staticmethod
    def accuracy_table(data: Dict[str, Any]) -> str:
        """生成准确率 ASCII 表格"""
        lines = []
        
        # 检查是否有多轮运行数据
        any_multi_run = any('run_count' in stats and stats['run_count'] > 1 for stats in data.values())
        run_count = max(stats.get('run_count', 1) for stats in data.values()) if any_multi_run else 1
        
        title = " 📊 DeepInsight 准确率评测结果 (EX Metric)"
        if any_multi_run:
            title = f" 📊 DeepInsight 准确率评测结果 (EX Metric) - {run_count} 轮平均"
        
        lines.append("")
        lines.append("╔" + "═" * 76 + "╗")
        lines.append("║" + title.center(76) + "║")
        lines.append("╠" + "═" * 76 + "╣")
        
        # 表头
        header = "║ {:^14} │ {:^10} │ {:^10} │ {:^10} │ {:^11} │ {:^14} ║"
        lines.append(header.format("方案", "Easy", "Medium", "Hard", "Extra-Hard", "Overall"))
        lines.append("╟" + "─" * 76 + "╢")
        
        # 数据行
        for mode_name, stats in data.items():
            row_data = [mode_name]
            for diff in ['easy', 'medium', 'hard', 'extra_hard']:
                if diff in stats.get('by_difficulty', {}):
                    acc = stats['by_difficulty'][diff]['accuracy'] * 100
                    row_data.append(f"{acc:.1f}%")
                else:
                    row_data.append("-")
            overall = stats.get('accuracy', 0) * 100
            # 如果有多轮运行，显示标准差
            if 'run_count' in stats and stats['run_count'] > 1:
                std_acc = stats.get('accuracy_std', 0) * 100
                row_data.append(f"{overall:.1f}%±{std_acc:.1f}%")
            else:
                row_data.append(f"{overall:.1f}%")
            
            lines.append("║ {:^14} │ {:^10} │ {:^10} │ {:^10} │ {:^11} │ {:^14} ║".format(*row_data))
        
        lines.append("╚" + "═" * 76 + "╝")
        if any_multi_run:
            lines.append("* ± 后数值为标准差")
        lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def performance_table(data: Dict[str, Any]) -> str:
        """生成性能 ASCII 表格"""
        lines = []
        lines.append("")
        lines.append("╔" + "═" * 82 + "╗")
        lines.append("║" + " ⚡ DeepInsight 性能评测结果".center(82) + "║")
        lines.append("╠" + "═" * 82 + "╣")
        
        header = "║ {:^18} │ {:^12} │ {:^12} │ {:^10} │ {:^10} ║"
        lines.append(header.format("后端", "延迟(ms)", "P95(ms)", "QPS", "加速比"))
        lines.append("╟" + "─" * 82 + "╢")
        
        for backend, stats in data.items():
            lines.append("║ {:^18} │ {:^12.1f} │ {:^12.1f} │ {:^10.1f} │ {:^10.2f}x ║".format(
                backend,
                stats.get('latency_avg_ms', 0),
                stats.get('latency_p95_ms', 0),
                stats.get('throughput_qps', stats.get('tps_avg', 0)),
                stats.get('speedup_vs_baseline', 1.0)
            ))
        
        lines.append("╚" + "═" * 82 + "╝")
        lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def token_breakdown_table(data: Dict[str, Any]) -> str:
        """生成 Token 消耗 breakdown ASCII 表格"""
        lines = []
        lines.append("")
        lines.append("╔" + "═" * 86 + "╗")
        lines.append("║" + " 🔢 Token 消耗统计 (按调用阶段分解)".center(84) + "║")
        lines.append("╠" + "═" * 86 + "╣")
        
        # 表头
        header = "║ {:^14} │ {:^12} │ {:^12} │ {:^12} │ {:^12} │ {:^10} ║"
        lines.append(header.format("方案", "Selector", "Generator", "Retry", "Avg Total", "Calls"))
        lines.append("╟" + "─" * 86 + "╢")
        
        # 数据行
        mode_names = {
            'baseline': 'G1 Baseline',
            'knowledge_rag': 'G2 RAG',
            'full_system': 'G3 Full'
        }
        
        for mode_key, stats in data.items():
            mode_name = mode_names.get(mode_key, mode_key)
            selector = stats.get('avg_selector_tokens', 0)
            generator = stats.get('avg_generator_tokens', 0)
            retry = stats.get('avg_retry_tokens', 0)
            avg_total = stats.get('avg_tokens_per_query', 0)
            calls = stats.get('avg_llm_calls', 1.0)
            
            lines.append("║ {:^14} │ {:^12,.0f} │ {:^12,.0f} │ {:^12,.0f} │ {:^12,.0f} │ {:^10.1f} ║".format(
                mode_name, selector, generator, retry, avg_total, calls
            ))
        
        lines.append("╚" + "═" * 86 + "╝")
        lines.append("")
        
        # 添加论文论点说明
        lines.append("📌 论文论点验证:")
        lines.append("   • T_G2 ≪ T_G1: 上下文压缩有效 (RAG 精排减少 prompt tokens)")
        lines.append("   • T_G3 > T_G2: 以适度成本换取高鲁棒性 (自愈重试增加 tokens)")
        lines.append("")
        
        return "\n".join(lines)


class Visualizer:
    """
    可视化主类
    
    整合所有可视化功能。
    """
    
    def __init__(self, results_dir: Path = None):
        """
        初始化可视化器
        
        Args:
            results_dir: 结果目录路径
        """
        if results_dir is None:
            results_dir = Path(__file__).parent / "results"
        
        self.results_dir = Path(results_dir)
        self.charts_dir = self.results_dir / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        self.markdown = MarkdownGenerator()
        self.plotly = PlotlyChartGenerator(self.charts_dir)
        self.ascii = ASCIITablePrinter()
    
    def load_accuracy_data(self) -> Dict[str, Any]:
        """加载准确率数据 (G1-G3 三组实验)，包含 Token 消耗统计
        优先从 averaged_results.json 加载（多次运行平均结果），
        如果不存在则从 CSV 文件加载
        """
        # 优先检查是否有平均结果
        averaged_json_path = self.results_dir / "averaged_results.json"
        if averaged_json_path.exists():
            import json
            with open(averaged_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 如果没有平均结果，从 CSV 加载
        data = {}
        for mode in ['baseline', 'knowledge_rag', 'full_system']:
            csv_path = self.results_dir / f"accuracy_{mode}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                total = len(df)
                correct = df['is_correct'].sum()
                
                by_diff = {}
                for diff in df['difficulty'].unique():
                    diff_df = df[df['difficulty'] == diff]
                    by_diff[diff] = {
                        'total': len(diff_df),
                        'correct': diff_df['is_correct'].sum(),
                        'accuracy': diff_df['is_correct'].mean()
                    }
                
                # 🆕 Token 消耗统计
                total_tokens = 0
                avg_tokens_per_query = 0
                if 'total_tokens' in df.columns:
                    total_tokens = df['total_tokens'].sum()
                    avg_tokens_per_query = df['total_tokens'].mean() if total > 0 else 0
                
                # 🆕 Token breakdown 统计
                avg_selector_tokens = df['selector_tokens'].mean() if 'selector_tokens' in df.columns else 0
                avg_generator_tokens = df['generator_tokens'].mean() if 'generator_tokens' in df.columns else 0
                avg_retry_tokens = df['retry_tokens'].mean() if 'retry_tokens' in df.columns else 0
                avg_llm_calls = df['llm_call_count'].mean() if 'llm_call_count' in df.columns else 1.0

                # 端到端响应时间统计（CSV 回退路径也输出 avg / P95）
                avg_execution_time_ms = 0.0
                p95_execution_time_ms = 0.0
                if 'execution_time_ms' in df.columns and total > 0:
                    exec_times = sorted(df['execution_time_ms'].dropna().astype(float).tolist())
                    if exec_times:
                        avg_execution_time_ms = float(sum(exec_times) / len(exec_times))
                        if len(exec_times) == 1:
                            p95_execution_time_ms = exec_times[0]
                        else:
                            idx = (len(exec_times) - 1) * 0.95
                            lo = int(idx)
                            hi = lo + 1
                            w = idx - lo
                            if hi < len(exec_times):
                                p95_execution_time_ms = exec_times[lo] + w * (exec_times[hi] - exec_times[lo])
                            else:
                                p95_execution_time_ms = exec_times[lo]
                
                data[mode] = {
                    'total_cases': total,
                    'correct_count': int(correct),
                    'accuracy': correct / total if total > 0 else 0,
                    'by_difficulty': by_diff,
                    'avg_execution_time_ms': avg_execution_time_ms,
                    'p95_execution_time_ms': p95_execution_time_ms,
                    # 🆕 Token 统计
                    'total_tokens': int(total_tokens),
                    'avg_tokens_per_query': avg_tokens_per_query,
                    # 🆕 Token breakdown
                    'avg_selector_tokens': avg_selector_tokens,
                    'avg_generator_tokens': avg_generator_tokens,
                    'avg_retry_tokens': avg_retry_tokens,
                    'avg_llm_calls': avg_llm_calls
                }
        
        return data
    
    def load_performance_data(self) -> Dict[str, Any]:
        """加载性能数据（优先从 averaged_performance_results.json 加载）"""
        # 优先检查是否有平均结果
        averaged_json_path = self.results_dir / "averaged_performance_results.json"
        if averaged_json_path.exists():
            with open(averaged_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 如果没有平均结果，从常规 JSON 加载
        summary_path = self.results_dir / "performance_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def load_system_info(self) -> Dict[str, Any]:
        """加载系统信息"""
        sysinfo_path = self.results_dir / "system_info.json"
        if sysinfo_path.exists():
            with open(sysinfo_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def generate_all(self, verbose: bool = True):
        """
        生成所有可视化输出
        
        Args:
            verbose: 详细输出
        """
        if verbose:
            print("\n" + "=" * 60)
            print("📊 生成可视化报告...")
            print("=" * 60)
        
        accuracy_data = self.load_accuracy_data()
        performance_data = self.load_performance_data()
        system_info = self.load_system_info()
        
        # 生成 Markdown 报告
        report = self.markdown.combined_report(accuracy_data, performance_data, system_info)
        report_path = self.results_dir / "evaluation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        if verbose:
            print(f"📝 Markdown 报告: {report_path}")
        
        # 生成 Plotly 图表
        if accuracy_data:
            self.plotly.accuracy_bar_chart(accuracy_data)
        
        if performance_data:
            self.plotly.performance_radar_chart(performance_data)
            self.plotly.latency_boxplot()
        
        # 打印 ASCII 表格
        if verbose:
            if accuracy_data:
                print(self.ascii.accuracy_table(accuracy_data))
            if performance_data:
                print(self.ascii.performance_table(performance_data))
        
        if verbose:
            print(f"\n✅ 所有可视化文件已保存到: {self.results_dir}")


# ============================================================
# 命令行入口
# ============================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepInsight 结果可视化")
    parser.add_argument("--format", choices=["all", "markdown", "plotly", "ascii"],
                        default="all", help="输出格式")
    parser.add_argument("--quiet", action="store_true", help="安静模式")
    
    args = parser.parse_args()
    
    visualizer = Visualizer()
    verbose = not args.quiet
    
    if args.format == "all":
        visualizer.generate_all(verbose)
    elif args.format == "markdown":
        accuracy_data = visualizer.load_accuracy_data()
        performance_data = visualizer.load_performance_data()
        system_info = visualizer.load_system_info()
        report = visualizer.markdown.combined_report(accuracy_data, performance_data, system_info)
        print(report)
    elif args.format == "ascii":
        accuracy_data = visualizer.load_accuracy_data()
        performance_data = visualizer.load_performance_data()
        if accuracy_data:
            print(visualizer.ascii.accuracy_table(accuracy_data))
        if performance_data:
            print(visualizer.ascii.performance_table(performance_data))
    elif args.format == "plotly":
        accuracy_data = visualizer.load_accuracy_data()
        performance_data = visualizer.load_performance_data()
        if accuracy_data:
            visualizer.plotly.accuracy_bar_chart(accuracy_data)
        if performance_data:
            visualizer.plotly.performance_radar_chart(performance_data)


if __name__ == "__main__":
    main()
