"""
性能评测模块 (Performance Benchmark)
====================================

真实对比两种推理后端的性能：
1. PyTorch FP32  - 原生 HuggingFace 模型 (基准)
2. OpenVINO FP32 - OpenVINO 优化模型

评测指标（全部为真实测量）：
- Latency (ms): Embedding 推理延迟
- Throughput (queries/s): 每秒处理查询数
- Memory (MB): 模型内存占用

学术意义:
    量化 OpenVINO 对 Embedding 模型的优化效果，
    为边缘部署场景提供真实性能参考数据。
"""

import sys
import os
import time
import json
import gc
import psutil
import statistics
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import argparse
import traceback

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# === 路径处理 ===
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 尝试导入 OpenVINO
try:
    from optimum.intel import OVModelForFeatureExtraction
    OPENVINO_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 警告: 无法导入 OpenVINO 模块 - {e}")
    OPENVINO_AVAILABLE = False


class InferenceBackend(Enum):
    """推理后端枚举"""
    PYTORCH_FP32 = "pytorch_fp32"
    OPENVINO_FP32 = "openvino_fp32"


@dataclass
class PerformanceMetrics:
    """
    单次推理的性能指标
    
    Attributes:
        backend: 推理后端
        query: 测试查询
        latency_ms: 推理延迟 (毫秒)
        timestamp: 测试时间戳
    """
    backend: str
    query: str
    latency_ms: float
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BackendSummary:
    """
    后端性能摘要
    
    Attributes:
        backend: 推理后端名称
        sample_count: 采样次数
        latency_avg_ms: 平均延迟
        latency_p50_ms: P50 延迟
        latency_p95_ms: P95 延迟
        latency_min_ms: 最小延迟
        latency_max_ms: 最大延迟
        throughput_qps: 吞吐量 (queries/s)
        speedup_vs_baseline: 相对基准的加速比
        run_count: 运行次数（多轮运行时）
        latency_std_ms: 延迟标准差（多轮运行时）
    """
    backend: str
    sample_count: int
    latency_avg_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_min_ms: float
    latency_max_ms: float
    throughput_qps: float
    speedup_vs_baseline: float = 1.0
    run_count: int = 1
    latency_std_ms: float = 0.0


@dataclass
class SystemInfo:
    """
    系统硬件信息
    
    用于论文中描述实验环境。
    """
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    memory_total_gb: float
    gpu_info: str
    os_info: str
    python_version: str
    torch_version: str
    openvino_version: str = ""


class PerformanceBenchmark:
    """
    性能评测主类
    
    管理 PyTorch vs OpenVINO 的真实性能对比测试。
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化性能评测器
        
        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            config_path = PROJECT_ROOT / "data" / "config.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # 模型路径
        self.pytorch_model_path = PROJECT_ROOT / "models" / "bge-small-pt"
        self.openvino_model_path = PROJECT_ROOT / "models" / "bge-small-ov"
        
        # 结果目录
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 性能数据存储
        self.metrics: Dict[InferenceBackend, List[PerformanceMetrics]] = {
            backend: [] for backend in InferenceBackend
        }
        
        # 缓存加载的模型
        self._models: Dict[InferenceBackend, Any] = {}
        self._tokenizers: Dict[InferenceBackend, Any] = {}
        
        # 加载测试查询集
        self.test_queries = self._load_test_queries()
        
        # 获取系统信息
        self.system_info = self._collect_system_info()
        
        print("✅ PerformanceBenchmark 初始化完成")
        self._print_system_info()
    
    def _load_test_queries(self) -> List[str]:
        """从 evaluation_set_adventureworks.json 加载测试查询"""
        eval_file = Path(__file__).parent / "evaluation_set_adventureworks.json"
        if eval_file.exists():
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                queries = [item["question"] for item in data if "question" in item]
                print(f"📋 已加载 {len(queries)} 道测试查询")
                return queries
            except Exception as e:
                print(f"⚠️ 加载测试查询失败: {e}")
        
        # 如果加载失败，使用默认查询集
        print("⚠️ 使用默认测试查询集")
        return [
            "统计订单总数",
            "查询所有产品",
            "获取客户列表",
            "每个区域的总销售额是多少",
            "找出购买次数超过5次的客户",
            "按月份统计2017年的订单数量",
            "找出每个类别中销售额最高的产品，并显示其供应商信息",
            "计算每个客户最近一年的消费金额并按降序排列",
            "构建客户价值分层，基于购买频率和消费金额进行RFM分析"
        ]
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _collect_system_info(self) -> SystemInfo:
        """收集系统硬件信息"""
        import platform
        
        cpu_info = {
            "model": platform.processor() or "Unknown",
            "cores": psutil.cpu_count(logical=False) or 0,
            "threads": psutil.cpu_count(logical=True) or 0
        }
        
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        # 检测 GPU
        gpu_info = "No GPU detected"
        try:
            if torch.cuda.is_available():
                gpu_info = torch.cuda.get_device_name(0)
        except Exception:
            pass
        
        # OpenVINO 版本
        openvino_version = ""
        try:
            import openvino
            openvino_version = openvino.__version__
        except ImportError:
            openvino_version = "Not installed"
        
        return SystemInfo(
            cpu_model=cpu_info["model"],
            cpu_cores=cpu_info["cores"],
            cpu_threads=cpu_info["threads"],
            memory_total_gb=memory_gb,
            gpu_info=gpu_info,
            os_info=f"{platform.system()} {platform.release()}",
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            openvino_version=openvino_version
        )
    
    def _print_system_info(self):
        """打印系统信息"""
        info = self.system_info
        print("\n🖥️ 系统配置:")
        print(f"   CPU: {info.cpu_model}")
        print(f"   核心/线程: {info.cpu_cores} / {info.cpu_threads}")
        print(f"   内存: {info.memory_total_gb:.1f} GB")
        print(f"   GPU: {info.gpu_info}")
        print(f"   PyTorch: {info.torch_version}")
        print(f"   OpenVINO: {info.openvino_version}")
    
    def _load_pytorch_model(self) -> Tuple[Any, Any]:
        """
        加载 PyTorch 模型
        
        Returns:
            (model, tokenizer)
        """
        if InferenceBackend.PYTORCH_FP32 in self._models:
            return self._models[InferenceBackend.PYTORCH_FP32], self._tokenizers[InferenceBackend.PYTORCH_FP32]
        
        print(f"\n📦 加载 PyTorch 模型: {self.pytorch_model_path}")
        
        model = AutoModel.from_pretrained(str(self.pytorch_model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(self.pytorch_model_path))
        model.eval()  # 设为评估模式
        
        print(f"   ✅ 加载成功")
        
        self._models[InferenceBackend.PYTORCH_FP32] = model
        self._tokenizers[InferenceBackend.PYTORCH_FP32] = tokenizer
        
        return model, tokenizer
    
    def _load_openvino_model(self) -> Tuple[Any, Any]:
        """
        加载 OpenVINO 模型
        
        Returns:
            (model, tokenizer)
        """
        if InferenceBackend.OPENVINO_FP32 in self._models:
            return self._models[InferenceBackend.OPENVINO_FP32], self._tokenizers[InferenceBackend.OPENVINO_FP32]
        
        if not OPENVINO_AVAILABLE:
            raise RuntimeError("OpenVINO 不可用")
        
        print(f"\n📦 加载 OpenVINO 模型: {self.openvino_model_path}")
        
        model = OVModelForFeatureExtraction.from_pretrained(
            str(self.openvino_model_path),
            device="CPU",
            ov_config={"PERFORMANCE_HINT": "LATENCY"}
        )
        tokenizer = AutoTokenizer.from_pretrained(str(self.openvino_model_path))
        
        print(f"   ✅ 加载成功")
        
        self._models[InferenceBackend.OPENVINO_FP32] = model
        self._tokenizers[InferenceBackend.OPENVINO_FP32] = tokenizer
        
        return model, tokenizer
    
    def _run_pytorch_inference(self, query: str) -> float:
        """
        执行 PyTorch 推理并测量延迟（仅测量推理时间，不包含 Tokenization）
        
        Args:
            query: 输入查询文本
            
        Returns:
            延迟时间 (毫秒)
        """
        model, tokenizer = self._load_pytorch_model()
        
        # Tokenize（不计入推理延迟）
        inputs = tokenizer(
            query, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # 推理并计时（仅测量推理部分）
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model(**inputs)
            # 提取 CLS token embedding
            embedding = outputs.last_hidden_state[:, 0].squeeze().numpy()
        
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        return latency_ms
    
    def _run_openvino_inference(self, query: str) -> float:
        """
        执行 OpenVINO 推理并测量延迟（仅测量推理时间，不包含 Tokenization）
        
        Args:
            query: 输入查询文本
            
        Returns:
            延迟时间 (毫秒)
        """
        model, tokenizer = self._load_openvino_model()
        
        # Tokenize（不计入推理延迟）
        inputs = tokenizer(
            query, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # 推理并计时（仅测量推理部分）
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model(**inputs)
            # 提取 CLS token embedding
            embedding = outputs.last_hidden_state[:, 0].squeeze().numpy()
        
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        return latency_ms
    
    def _measure_inference(
        self, 
        backend: InferenceBackend, 
        query: str
    ) -> PerformanceMetrics:
        """
        测量单次推理性能
        
        Args:
            backend: 推理后端
            query: 测试查询
            
        Returns:
            性能指标
        """
        if backend == InferenceBackend.PYTORCH_FP32:
            latency_ms = self._run_pytorch_inference(query)
        elif backend == InferenceBackend.OPENVINO_FP32:
            latency_ms = self._run_openvino_inference(query)
        else:
            raise ValueError(f"未知后端: {backend}")
        
        return PerformanceMetrics(
            backend=backend.value,
            query=query,
            latency_ms=latency_ms
        )
    
    def _unload_model(self, backend: InferenceBackend):
        """
        卸载模型并清理内存
        
        Args:
            backend: 要卸载的后端
        """
        if backend in self._models:
            del self._models[backend]
        if backend in self._tokenizers:
            del self._tokenizers[backend]
        
        # 清理 PyTorch 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制垃圾回收
        gc.collect()
        
        print(f"   🗑️ 已卸载 {backend.value} 模型")
    
    def run_benchmark(
        self, 
        backend: InferenceBackend,
        queries: List[str] = None,
        samples_per_query: int = 5,
        warmup_runs: int = 3,
        verbose: bool = True
    ) -> BackendSummary:
        """
        运行单个后端的性能测试
        
        Args:
            backend: 推理后端
            queries: 测试查询列表，默认使用标准集
            samples_per_query: 每个查询的采样次数
            warmup_runs: 预热运行次数（每个查询）
            verbose: 详细输出
            
        Returns:
            后端性能摘要
        """
        if queries is None:
            queries = self.test_queries
        
        if verbose:
            print(f"\n🔬 测试后端: {backend.value}")
            print(f"   查询数: {len(queries)}, 采样次数: {samples_per_query}")
        
        # 预热（让模型/缓存稳定）- 每个查询都预热
        if warmup_runs > 0 and verbose:
            print(f"   🔥 预热中 ({warmup_runs} 次/查询)...")
        for query in queries:
            for _ in range(warmup_runs):
                self._measure_inference(backend, query)
        
        # 正式测试
        all_metrics = []
        total_samples = len(queries) * samples_per_query
        current = 0
        
        for query in queries:
            for _ in range(samples_per_query):
                current += 1
                if verbose and current % 10 == 0:
                    print(f"   进度: {current}/{total_samples}")
                
                metric = self._measure_inference(backend, query)
                all_metrics.append(metric)
                self.metrics[backend].append(metric)
        
        # 计算统计
        summary = self._compute_backend_summary(backend, all_metrics)
        
        if verbose:
            self._print_backend_summary(summary)
        
        return summary
    
    def _compute_backend_summary(
        self, 
        backend: InferenceBackend, 
        metrics: List[PerformanceMetrics]
    ) -> BackendSummary:
        """计算后端性能统计（使用标准百分位数计算）"""
        latencies = [m.latency_ms for m in metrics]
        
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        
        # 使用标准百分位数计算（线性插值）
        def calculate_percentile(data_sorted, percentile):
            """
            计算百分位数（使用 NumPy 的线性插值方法）
            
            Args:
                data_sorted: 已排序的数据列表
                percentile: 百分位数 (0-100)
            
            Returns:
                百分位数值
            """
            n = len(data_sorted)
            if n == 0:
                return 0
            if n == 1:
                return data_sorted[0]
            
            # 计算索引
            idx = (n - 1) * percentile / 100.0
            lower = int(idx)
            upper = lower + 1
            weight = idx - lower
            
            if upper >= n:
                return data_sorted[lower]
            
            return data_sorted[lower] + weight * (data_sorted[upper] - data_sorted[lower])
        
        # 计算百分位数
        p50 = calculate_percentile(latencies_sorted, 50)
        p95 = calculate_percentile(latencies_sorted, 95)
        
        avg_latency = statistics.mean(latencies) if latencies else 0
        throughput = 1000 / avg_latency if avg_latency > 0 else 0  # queries/s
        
        return BackendSummary(
            backend=backend.value,
            sample_count=n,
            latency_avg_ms=avg_latency,
            latency_p50_ms=p50 if n > 0 else 0,
            latency_p95_ms=p95 if n > 0 else 0,
            latency_min_ms=min(latencies) if latencies else 0,
            latency_max_ms=max(latencies) if latencies else 0,
            throughput_qps=throughput
        )
    
    def _print_backend_summary(self, summary: BackendSummary):
        """打印后端摘要"""
        print(f"\n📊 {summary.backend} 性能统计:")
        print(f"   延迟 (ms): avg={summary.latency_avg_ms:.2f}, "
              f"P50={summary.latency_p50_ms:.2f}, P95={summary.latency_p95_ms:.2f}")
        print(f"   吞吐量: {summary.throughput_qps:.1f} queries/s")
    
    def run_all(
        self, 
        samples_per_query: int = 5,
        verbose: bool = True
    ) -> Dict[str, BackendSummary]:
        """
        运行全部两个后端的测试
        
        Args:
            samples_per_query: 每个查询采样次数
            verbose: 详细输出
            
        Returns:
            各后端的性能摘要
        """
        summaries = {}
        
        # PyTorch 基准
        if verbose:
            print("\n" + "=" * 70)
            print("开始测试 PyTorch FP32 后端")
            print("=" * 70)
        summaries['pytorch_fp32'] = self.run_benchmark(
            InferenceBackend.PYTORCH_FP32,
            samples_per_query=samples_per_query,
            verbose=verbose
        )
        
        # 卸载 PyTorch 模型，清理内存
        self._unload_model(InferenceBackend.PYTORCH_FP32)
        
        # OpenVINO FP32
        if OPENVINO_AVAILABLE:
            if verbose:
                print("\n" + "=" * 70)
                print("开始测试 OpenVINO FP32 后端")
                print("=" * 70)
            summaries['openvino_fp32'] = self.run_benchmark(
                InferenceBackend.OPENVINO_FP32,
                samples_per_query=samples_per_query,
                verbose=verbose
            )
        else:
            print("\n⚠️ OpenVINO 不可用，跳过 OpenVINO 测试")
        
        # 计算加速比
        baseline_latency = summaries['pytorch_fp32'].latency_avg_ms
        for name, summary in summaries.items():
            if summary.latency_avg_ms > 0:
                summary.speedup_vs_baseline = baseline_latency / summary.latency_avg_ms
        
        if verbose:
            self._print_comparison(summaries)
        
        # 保存结果
        self._save_results(summaries)
        
        return summaries
    
    def _print_comparison(self, summaries: Dict[str, BackendSummary]):
        """打印性能对比结果"""
        print("\n" + "=" * 85)
        print("📊 性能评测对比结果 (Embedding 推理)")
        print("=" * 85)
        
        # ASCII 表格
        header = f"{'后端':<18} | {'延迟(ms)':<12} | {'P95(ms)':<12} | {'吞吐量':<12} | {'加速比':<8}"
        print(header)
        print("-" * 85)
        
        for name, summary in summaries.items():
            row = (f"{name:<18} | "
                   f"{summary.latency_avg_ms:>10.2f} | "
                   f"{summary.latency_p95_ms:>10.2f} | "
                   f"{summary.throughput_qps:>10.1f} | "
                   f"{summary.speedup_vs_baseline:>6.2f}x")
            print(row)
        
        print("=" * 85)
        
        # 性能提升说明
        pytorch = summaries.get('pytorch_fp32')
        openvino = summaries.get('openvino_fp32')
        
        if pytorch and openvino:
            speedup = pytorch.latency_avg_ms / openvino.latency_avg_ms if openvino.latency_avg_ms > 0 else 0
            print(f"\n✨ OpenVINO 相比 PyTorch 加速 {speedup:.2f}x")
    
    def _save_results(self, summaries: Dict[str, BackendSummary]):
        """保存性能测试结果"""
        # 保存摘要
        summary_data = {name: asdict(s) for name, s in summaries.items()}
        summary_path = self.results_dir / "performance_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        # 保存详细指标
        all_metrics = []
        for backend, metrics in self.metrics.items():
            all_metrics.extend([asdict(m) for m in metrics])
        
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            csv_path = self.results_dir / "performance_details.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 保存系统信息
        sysinfo_path = self.results_dir / "system_info.json"
        with open(sysinfo_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.system_info), f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存到: {self.results_dir}")


def generate_ascii_table(summaries: Dict[str, BackendSummary]) -> str:
    """
    生成符合论文格式的 ASCII 表格
    """
    lines = []
    lines.append("=" * 90)
    lines.append("Table: Performance Comparison of Embedding Inference Backends")
    lines.append("=" * 90)
    lines.append(f"{'Backend':<20} | {'Latency(ms)':<15} | {'P95(ms)':<12} | {'QPS':<12} | {'Speedup':<10}")
    lines.append("-" * 90)
    
    for name, summary in summaries.items():
        lines.append(
            f"{name:<20} | "
            f"{summary.latency_avg_ms:>13.2f} | "
            f"{summary.latency_p95_ms:>10.2f} | "
            f"{summary.throughput_qps:>10.1f} | "
            f"{summary.speedup_vs_baseline:>8.2f}x"
        )
    
    lines.append("=" * 90)
    lines.append("Note: Speedup is relative to PyTorch FP32 baseline")
    lines.append("      Latency measured as embedding generation time for text queries")
    
    return "\n".join(lines)


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="DeepInsight 性能评测 (Embedding)")
    parser.add_argument("--samples", type=int, default=5, help="每查询采样次数")
    parser.add_argument("--test-mode", action="store_true", help="测试模式（快速，每查询1次）")
    parser.add_argument("--quiet", action="store_true", help="安静模式")
    parser.add_argument("--backend", choices=["pytorch", "openvino", "all"],
                        default="all", help="测试的后端")
    
    args = parser.parse_args()
    
    samples = 1 if args.test_mode else args.samples
    verbose = not args.quiet
    
    benchmark = PerformanceBenchmark()
    
    if args.backend == "all":
        summaries = benchmark.run_all(samples_per_query=samples, verbose=verbose)
        print("\n" + generate_ascii_table(summaries))
    else:
        backend_map = {
            "pytorch": InferenceBackend.PYTORCH_FP32,
            "openvino": InferenceBackend.OPENVINO_FP32
        }
        benchmark.run_benchmark(
            backend_map[args.backend],
            samples_per_query=samples,
            verbose=verbose
        )


if __name__ == "__main__":
    main()
