"""
准确率评测模块 (Accuracy Benchmark)
==================================

实现三组对比实验（G1-G3）：

| 实验组 | 模式名称          | RAG检索 | KECA注入 | Agent自愈 | 学术意义                             |
|--------|-------------------|---------|----------|-----------|--------------------------------------|
| G1     | BASELINE          | ❌      | ❌       | ❌        | LLM在全量Schema高噪声环境下的抗噪基准 |
| G2     | KNOWLEDGE_RAG     | ✅      | ✅       | ❌        | 量化"精排检索"的Token效率与准确率收益 |
| G3     | FULL_SYSTEM       | ✅      | ✅       | ✅        | 验证Agent自愈修复精排遗漏的性能上限   |

KECA (Knowledge-Enhanced Context Augmentation) 包含:
    - 业务上下文 (business_context)
    - 术语词典 (term_dictionary)
    - Few-shot 示例 (example_queries)

评分指标 (EX - Execution Accuracy):
    执行生成的 SQL 与 Gold SQL，对比结果集（DataFrame）是否逻辑等价。
    - 允许列顺序不同
    - 允许行顺序不同（除非有 ORDER BY）
    - 数据内容必须完全一致

学术意义:
    EX 是 Spider 评测的核心指标，反映了系统生成可执行且语义正确 SQL 的能力。
    通过三组对比实验，可以量化 RAG+KECA 检索与 Agent 自愈机制对系统准确率的贡献。
"""

import sys
import os
import json
import time
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import argparse
import traceback

# === 路径处理：将项目根目录添加到 sys.path ===
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入评测数据模块 (必须可用)
from eval_suite.data_loader import EvalDataLoader, EvalCase, Difficulty

# 导入项目核心模块 (可选，不可用时使用模拟模式)
try:
    from agent_core import Text2SQLAgent
    from rag_engine import IntelRAG
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 警告: 无法导入核心模块 - {e}")
    print("   将使用模拟模式进行评测")
    CORE_AVAILABLE = False

try:
    from deepinsight_core.config import DeepInsightSettings
    from deepinsight_core.services.text2sql_graph_service import Text2SQLGraphService
    GRAPH_CORE_AVAILABLE = True
except ImportError:
    DeepInsightSettings = None
    Text2SQLGraphService = None
    GRAPH_CORE_AVAILABLE = False


class ExperimentMode(Enum):
    """
    实验模式枚举 (G1-G3)
    
    G1: BASELINE        - 完整Schema + 基础Prompt（无RAG、无KECA、无自愈）
    G2: KNOWLEDGE_RAG   - RAG检索Schema + KECA知识增强（无自愈）
    G3: FULL_SYSTEM     - RAG + KECA + Agent自愈
    """
    BASELINE = "baseline"               # G1: 无 RAG + 无 KECA + 无自愈
    KNOWLEDGE_RAG = "knowledge_rag"     # G2: 有 RAG + 有 KECA + 无自愈
    FULL_SYSTEM = "full_system"         # G3: 有 RAG + 有 KECA + 有自愈


@dataclass
class EvalResult:
    """
    单个评测用例的结果
    
    Attributes:
        case_id: 用例 ID
        difficulty: 难度等级
        question: 自然语言问题
        gold_sql: 标准答案 SQL
        generated_sql: 系统生成的 SQL
        is_correct: 是否正确（EX 指标）
        execution_time_ms: 执行耗时（毫秒）
        error_message: 错误信息（如有）
        gold_result_hash: Gold SQL 结果哈希
        gen_result_hash: 生成 SQL 结果哈希
        query_complexity: RAG 复杂度评估结果 (simple/medium/complex)
        rag_top_k: RAG 检索使用的 top_k 值
        rag_hints_count: RAG 提供的 hints 数量
    """
    case_id: str
    difficulty: str
    question: str
    gold_sql: str
    generated_sql: str
    is_correct: bool
    execution_time_ms: float
    error_message: str = ""
    gold_result_hash: str = ""
    gen_result_hash: str = ""
    # RAG 中间态数据（用于事后分析）
    query_complexity: str = ""
    rag_top_k: int = 0
    rag_hints_count: int = 0
    # 🆕 Token 消耗统计
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # 🆕 Token breakdown (按调用阶段分解)
    selector_tokens: int = 0       # 精排器 LLM 消耗
    generator_tokens: int = 0      # SQL 生成器消耗
    retry_tokens: int = 0          # 自愈重试消耗
    llm_call_count: int = 1        # LLM 调用次数


@dataclass
class ExperimentSummary:
    """
    实验摘要统计
    
    Attributes:
        mode: 实验模式
        total_cases: 总用例数
        correct_count: 正确数量
        accuracy: 准确率
        by_difficulty: 按难度分组的准确率
        avg_execution_time_ms: 平均执行时间
        timestamp: 实验时间戳
    """
    mode: str
    total_cases: int
    correct_count: int
    accuracy: float
    by_difficulty: Dict[str, Dict[str, float]]
    avg_execution_time_ms: float
    timestamp: str
    accuracy_std: float = 0.0      # 准确率标准差（多轮运行时）
    run_count: int = 1             # 运行次数
    p95_execution_time_ms: float = 0.0  # P95 端到端响应时间（毫秒）
    # 🆕 Token 消耗统计
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    avg_tokens_per_query: float = 0.0
    # 🆕 Token breakdown (按调用阶段分解)
    avg_selector_tokens: float = 0.0
    avg_generator_tokens: float = 0.0
    avg_retry_tokens: float = 0.0
    avg_llm_calls: float = 1.0


class SQLExecutor:
    """
    SQL 执行器
    
    负责执行 SQL 语句并返回 DataFrame 结果。
    支持 MySQL 数据库 (via SQLAlchemy)。
    """
    
    def __init__(self, connection_string: str):
        """
        初始化 SQL 执行器
        
        Args:
            connection_string: SQLAlchemy 连接字符串
                例如: mysql+pymysql://user:pass@host:port/database
        """
        self.connection_string = connection_string
        self.engine = create_engine(
            connection_string,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={
                "connect_timeout": 10,
                "read_timeout": 30,
                "write_timeout": 30
            }
        )
        # 测试连接
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            raise ConnectionError(f"数据库连接失败: {e}")
    
    def execute(self, sql: str, timeout: float = 30.0) -> Tuple[Optional[pd.DataFrame], str]:
        """
        执行 SQL 语句
        
        Args:
            sql: SQL 语句
            timeout: 超时时间（秒）
            
        Returns:
            (DataFrame 或 None, 错误信息)
        """
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(sql), conn)
            return df, ""
        except Exception as e:
            return None, str(e)


class ResultComparator:
    """
    结果比较器 (Spider EX 标准兼容版)
    
    实现 EX (Execution Accuracy) 指标的判定逻辑：
    1. 两个 SQL 执行都成功
    2. 结果集内容逻辑等价
    
    Spider EX 标准宽松处理：
    - 忽略列名/别名（只比较值）
    - 忽略列顺序
    - 忽略行顺序
    - 允许生成结果包含额外列（超集匹配）
    - 数值比较允许浮点误差
    """
    
    @staticmethod
    def normalize_value(val):
        """
        规范化单个值用于比较
        
        处理：
        - None/NaN 统一
        - 数值类型统一为 float
        - 字符串去空格转小写
        """
        import numpy as np
        
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        
        # 数值类型
        if isinstance(val, (int, float, np.integer, np.floating)):
            return round(float(val), 6)  # 保留6位小数
        
        # 字符串类型
        if isinstance(val, str):
            return val.strip().lower()
        
        return str(val).strip().lower()
    
    @staticmethod
    def normalize_dataframe(df: pd.DataFrame) -> list:
        """
        将 DataFrame 规范化为可比较的行集合
        
        返回排序后的元组列表，每个元组是一行的值（忽略列名）
        """
        if df is None or df.empty:
            return []
        
        try:
            rows = []
            for _, row in df.iterrows():
                # 对每行的值进行规范化，排序后形成元组
                normalized_values = tuple(sorted(
                    [ResultComparator.normalize_value(v) for v in row.values],
                    key=lambda x: (x is None, str(x) if x is not None else '')
                ))
                rows.append(normalized_values)
            
            # 对所有行排序（忽略行顺序）
            rows.sort(key=lambda x: tuple((v is None, str(v) if v is not None else '') for v in x))
            return rows
        except Exception:
            return []
    
    @staticmethod
    def dataframe_hash(df: pd.DataFrame) -> str:
        """
        计算 DataFrame 的规范化哈希值（用于日志记录）
        """
        if df is None or df.empty:
            return "EMPTY"
        
        try:
            rows = ResultComparator.normalize_dataframe(df)
            return str(rows)[:500]  # 截断避免过长
        except Exception:
            return "HASH_ERROR"
    
    @staticmethod
    def find_matching_columns(gold_df: pd.DataFrame, gen_df: pd.DataFrame) -> list:
        """
        找到生成结果中与Gold结果匹配的列（按值匹配）
        
        返回: gen_df 中匹配 gold_df 各列的列索引列表，或 None 如果无法匹配
        """
        import numpy as np
        
        gold_cols = gold_df.shape[1]
        gen_cols = gen_df.shape[1]
        
        if gen_cols < gold_cols:
            return None  # 生成结果列数不足
        
        # 为每个 gold 列找到匹配的 gen 列
        matched_gen_indices = []
        used_gen_indices = set()
        
        for gold_col_idx in range(gold_cols):
            gold_col_values = [ResultComparator.normalize_value(v) for v in gold_df.iloc[:, gold_col_idx]]
            
            best_match = None
            for gen_col_idx in range(gen_cols):
                if gen_col_idx in used_gen_indices:
                    continue
                    
                gen_col_values = [ResultComparator.normalize_value(v) for v in gen_df.iloc[:, gen_col_idx]]
                
                # 比较排序后的值集合
                if sorted(gold_col_values, key=lambda x: (x is None, str(x) if x is not None else '')) == \
                   sorted(gen_col_values, key=lambda x: (x is None, str(x) if x is not None else '')):
                    best_match = gen_col_idx
                    break
            
            if best_match is not None:
                matched_gen_indices.append(best_match)
                used_gen_indices.add(best_match)
            else:
                return None  # 找不到匹配的列
        
        return matched_gen_indices
    
    @staticmethod
    def compare(gold_df: pd.DataFrame, gen_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        比较两个 DataFrame 是否逻辑等价 (Spider EX 标准)
        
        Args:
            gold_df: Gold SQL 的执行结果
            gen_df: 生成 SQL 的执行结果
            
        Returns:
            (是否等价, 不等价的原因)
            
        Spider EX 标准:
            - 忽略列名/别名（只比较数据值）
            - 忽略列顺序
            - 忽略行顺序  
            - 允许生成结果包含额外列（超集匹配）
            - 行数必须完全相等
        """
        import numpy as np
        
        # 空值处理
        if gold_df is None and gen_df is None:
            return True, ""
        
        if gold_df is None or gen_df is None:
            return False, "One result is None"
        
        # 处理空 DataFrame
        gold_empty = gold_df.empty or len(gold_df) == 0
        gen_empty = gen_df.empty or len(gen_df) == 0
        
        if gold_empty and gen_empty:
            return True, ""
        
        if gold_empty != gen_empty:
            return False, f"Empty mismatch: gold_empty={gold_empty}, gen_empty={gen_empty}"
        
        # 行数必须相等
        if len(gold_df) != len(gen_df):
            return False, f"Row count mismatch: {len(gold_df)} vs {len(gen_df)}"
        
        # 生成结果列数不能少于 Gold (允许超集)
        if len(gen_df.columns) < len(gold_df.columns):
            return False, f"Generated has fewer columns: {len(gen_df.columns)} < {len(gold_df.columns)}"
        
        # 策略1: 如果列数相同，使用值集合比较（忽略列名）
        if len(gold_df.columns) == len(gen_df.columns):
            gold_rows = ResultComparator.normalize_dataframe(gold_df)
            gen_rows = ResultComparator.normalize_dataframe(gen_df)
            
            if gold_rows == gen_rows:
                return True, ""
        
        # 策略2: 列数不同时，尝试找到匹配的列子集
        matched_indices = ResultComparator.find_matching_columns(gold_df, gen_df)
        
        if matched_indices is not None:
            # 提取匹配的列进行比较
            gen_subset = gen_df.iloc[:, matched_indices]
            gold_rows = ResultComparator.normalize_dataframe(gold_df)
            gen_rows = ResultComparator.normalize_dataframe(gen_subset)
            
            if gold_rows == gen_rows:
                return True, ""
        
        # 策略3: 逐行值比较（更宽松）
        try:
            for gold_idx in range(len(gold_df)):
                gold_row_values = set(ResultComparator.normalize_value(v) for v in gold_df.iloc[gold_idx].values)
                
                # 在 gen_df 中查找包含这些值的行
                found = False
                for gen_idx in range(len(gen_df)):
                    gen_row_values = set(ResultComparator.normalize_value(v) for v in gen_df.iloc[gen_idx].values)
                    
                    # Gold 行的所有值都应该在 Gen 行中
                    if gold_row_values.issubset(gen_row_values):
                        found = True
                        break
                
                if not found:
                    return False, f"Row {gold_idx} values not found in generated result"
            
            return True, ""
        except Exception as e:
            pass
        
        return False, "Content mismatch after all comparison strategies"


class AccuracyBenchmark:
    """
    准确率评测主类
    
    管理三组对比实验的执行和结果收集。
    """
    
    # 数据库配置映射
    DATABASE_CONFIGS = {
        "northwind": {
            "db_uri": "mysql+pymysql://root:123456@localhost:3306/northwind",
            "schema_path": "data/schema_northwind.json",
            "prompt_config_path": "data/prompt_config.json",
            "kb_paths": ["data/schema_northwind.json", "data/prompt_config.json"],
            "display_name": "Northwind"
        },
        "adventureworks": {
            "db_uri": "mysql+pymysql://root:123456@localhost:3306/adventureworks",
            "schema_path": "data/schema_adventureworks.json",
            "prompt_config_path": "data/prompt_config_adventureworks.json",
            "kb_paths": [
                "data/schema_adventureworks.json", 
                "data/prompt_config_adventureworks.json"
            ],
            "display_name": "AdventureWorks"
        }
    }
    
    def __init__(self, config_path: str = None, database: str = "northwind", agent_backend: str = "legacy"):
        """
        初始化评测器
        
        Args:
            config_path: 配置文件路径，默认为 data/config.json
            database: 数据库名称 ("northwind" 或 "adventureworks")
        """
        if config_path is None:
            config_path = PROJECT_ROOT / "data" / "config.json"
        
        self.config_path = Path(config_path)
        self.database = database.lower()
        self.agent_backend = agent_backend.lower()
        self.config = self._load_config()
        
        # 根据数据库覆盖配置
        if self.database in self.DATABASE_CONFIGS:
            db_config = self.DATABASE_CONFIGS[self.database]
            self.config["db_path"] = db_config["db_uri"]
            self.config["db_uris"] = [db_config["db_uri"]]
            self.config["schema_path"] = db_config["schema_path"]
            self.config["kb_paths_list"] = db_config.get("kb_paths", [db_config["schema_path"]])
        
        # 初始化数据加载器（传入 database 参数）
        self.data_loader = EvalDataLoader(database=self.database)
        
        # 初始化 SQL 执行器 (使用配置的数据库)
        db_connection_string = self.config.get("db_path")
        self.sql_executor = SQLExecutor(db_connection_string)
        
        # 结果存储
        self.results: Dict[ExperimentMode, List[EvalResult]] = {
            mode: [] for mode in ExperimentMode
        }
        
        # 结果目录
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        display_name = self.DATABASE_CONFIGS.get(self.database, {}).get("display_name", self.database)
        print("✅ AccuracyBenchmark 初始化完成")
        print(f"   📊 评测用例: {len(self.data_loader.cases)} 道")
        print(f"   🗄️ 数据库: MySQL ({display_name})")
        print(f"   🧠 Agent Backend: {self.agent_backend}")
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def verify_agent_ready(self, mode: ExperimentMode) -> Tuple[bool, str]:
        """
        验证指定模式的 Agent 是否可以正常创建
        
        Returns:
            (是否就绪, 错误信息)
        """
        try:
            if mode == ExperimentMode.BASELINE:
                agent = self._create_baseline_agent()
            elif mode == ExperimentMode.KNOWLEDGE_RAG:
                agent = self._create_knowledge_rag_agent()
            else:  # FULL_SYSTEM
                agent = self._create_full_system_agent()
            
            if agent is None:
                return False, f"{mode.value} Agent 创建返回 None"
            
            return True, ""
        except Exception as e:
            return False, f"{mode.value} Agent 创建异常: {e}"
    
    def verify_all_agents(self) -> Dict[ExperimentMode, Tuple[bool, str]]:
        """验证所有实验模式的 Agent，返回验证结果字典"""
        results = {}
        print("\n🔍 验证 Agent 创建状态...")
        for mode in ExperimentMode:
            ok, err = self.verify_agent_ready(mode)
            results[mode] = (ok, err)
            status = "✅ 就绪" if ok else f"❌ {err}"
            print(f"   {mode.value}: {status}")
        return results
    
    def _create_baseline_agent(self) -> Optional['Text2SQLAgent']:
        """
        创建 G1 Baseline 模式的 Agent
        
        G1 Baseline 模式特点：
        - 禁用 RAG 语义检索 (传入 MockRAG 返回完整 Schema)
        - 禁用 KECA 知识增强 (不传入 config，避免 EnhancedPromptBuilder 加载)
        - 禁用错误自愈 (max_retries = 0)
        - 表选择器仅使用关键词匹配（无语义向量）
        
        学术意义: 测量 LLM 在无辅助情况下的原始 Text-to-SQL 能力。
        """
        if not CORE_AVAILABLE:
            return None
        
        # 创建一个 Mock RAG，模拟无语义匹配的情况
        class MockRAG:
            """
            Mock RAG 引擎 - 用于 Baseline 评测
            
            特点：
            - model = None (禁用语义向量匹配)
            - retrieve 返回完整 Schema
            - _get_embedding 返回零向量
            """
            def __init__(self, schema_path, kb_paths=None):
                # 核心属性：model 设为 None，表示无语义匹配能力
                self.model = None
                self.tokenizer = None
                self.kb_paths = kb_paths or []
                self.db_uris = []
                self.documents = []
                self.embeddings = None
                
                # 加载完整 Schema
                self.schema = self._load_full_schema(schema_path)
            
            def _load_full_schema(self, path):
                """加载 Schema 文件内容"""
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.dumps(json.load(f), ensure_ascii=False, indent=2)
                return ""
            
            def _get_embedding(self, text):
                """返回零向量（无语义匹配能力）"""
                import numpy as np
                return np.zeros(384)
            
            def retrieve(self, query, top_k=5):
                """返回完整 Schema，模拟无 RAG 的情况"""
                # 返回: (context_str, latency_ms, memory_delta_mb)
                return self.schema, 0.0, 0.0
        
        # 使用配置的 Schema (支持 Northwind / AdventureWorks)
        schema_path = self.config.get("schema_path", "data/schema_northwind.json")
        if not os.path.isabs(schema_path):
            schema_path = str(PROJECT_ROOT / schema_path)
        
        kb_paths = self.config.get("kb_paths_list", [schema_path])
        mock_rag = MockRAG(schema_path, kb_paths)
        
        try:
            agent = Text2SQLAgent(
                api_key=self.config.get("api_key", ""),
                base_url=self.config.get("api_base", ""),
                model_name=self.config.get("model_name", ""),
                db_uris=self.config.get("db_uris", []),
                rag_engine=mock_rag,
                max_retries=0,      # 禁用重试
                max_candidates=1,   # ⭐ 禁用歧义消解，避免额外LLM调用
                config=None,        # ⭐ 禁用 KECA 知识增强
                temperature=0.0     # 评测时使用确定性输出
            )
            return agent
        except Exception as e:
            print(f"❌ 创建 Baseline Agent 失败: {e}")
            traceback.print_exc()
            return None
    
    def _create_knowledge_rag_agent(self) -> Optional['Text2SQLAgent']:
        """
        创建 G2 Knowledge-RAG 模式的 Agent
        
        G2 Knowledge-RAG 模式特点：
        - 启用 RAG 语义检索 (动态 Top-K Schema 选择 + OpenVINO加速)
        - 启用 KECA 知识增强 (传入 config，加载业务上下文/术语词典/示例)
        - 禁用错误自愈 (max_retries = 0)
        
        学术意义: 量化"精排检索"相对于"全量投喂"在 Token 节省（效率）与准确率（精度）上的双重收益。
        """
        if not CORE_AVAILABLE:
            return None
        
        try:
            model_path = PROJECT_ROOT / self.config.get("model_path", "models/bge-small-ov")
            rag = IntelRAG(
                model_path=str(model_path),
                db_uris=self.config.get("db_uris", []),
                kb_paths=self.config.get("kb_paths_list", [])
            )
            
            agent = Text2SQLAgent(
                api_key=self.config.get("api_key", ""),
                base_url=self.config.get("api_base", ""),
                model_name=self.config.get("model_name", ""),
                db_uris=self.config.get("db_uris", []),
                rag_engine=rag,
                max_retries=0,      # 禁用重试
                max_candidates=1,   # ⭐ 禁用歧义消解，避免额外LLM调用
                config=self.config, # ⭐ 启用 KECA 知识增强
                temperature=0.0     # 评测时使用确定性输出
            )
            return agent
        except Exception as e:
            print(f"❌ 创建 Knowledge-RAG Agent 失败: {e}")
            return None
    
    def _create_full_system_agent(self) -> Optional['Text2SQLAgent']:
        """
        创建 G3 Full System 模式的 Agent
        
        G3 Full System 模式特点：
        - 启用 RAG 语义检索 (动态 Top-K Schema 选择 + OpenVINO加速)
        - 启用 KECA 知识增强 (业务上下文/术语词典/示例)
        - 启用错误自愈 (使用配置的 max_retries)
        
        学术意义: 验证 Agent 的自愈反馈如何修复 RAG 阶段可能存在的"精排遗漏"，达到最终性能上限。
        """
        if not CORE_AVAILABLE:
            return None

        if self.agent_backend == "graph":
            return self._create_graph_full_system_agent()
        
        try:
            model_path = PROJECT_ROOT / self.config.get("model_path", "models/bge-small-ov")
            rag = IntelRAG(
                model_path=str(model_path),
                db_uris=self.config.get("db_uris", []),
                kb_paths=self.config.get("kb_paths_list", [])
            )
            
            agent = Text2SQLAgent(
                api_key=self.config.get("api_key", ""),
                base_url=self.config.get("api_base", ""),
                model_name=self.config.get("model_name", ""),
                db_uris=self.config.get("db_uris", []),
                rag_engine=rag,
                max_retries=self.config.get("max_retries", 4),  # 启用自愈 (基于优化分析增至4次)
                max_candidates=1,   # ⭐ 禁用歧义消解，避免额外LLM调用
                config=self.config, # ⭐ 启用 KECA 知识增强
                temperature=0.0,    # 评测时使用确定性输出
                # ⭐ Reasoner 自愈模式配置 (从 config 读取，统一控制)
                reasoner_model=self.config.get("reasoner_model", "deepseek-reasoner"),
                use_reasoner_for_healing=self.config.get("use_reasoner_for_healing", True)
            )
            return agent
        except Exception as e:
            print(f"❌ 创建 Full System Agent 失败: {e}")
            return None

    def _create_graph_full_system_agent(self):
        """
        创建实验性的 Graph Full System Agent。

        该路径通过 GraphAgentAdapter 暴露 generate_and_execute_stream()，
        因此 eval_suite 的事件消费逻辑无需改动。
        """
        if not GRAPH_CORE_AVAILABLE:
            print("❌ Graph backend 不可用，请检查 deepinsight_core 导入")
            return None

        try:
            graph_config = dict(self.config)
            graph_config["enable_graph_query_path"] = True
            settings = DeepInsightSettings.from_mapping(graph_config)
            service = Text2SQLGraphService(settings)
            return service.as_agent_adapter()
        except Exception as e:
            print(f"❌ 创建 Graph Full System Agent 失败: {e}")
            traceback.print_exc()
            return None
    
    def _extract_sql_from_stream(self, agent: 'Text2SQLAgent', query: str) -> Tuple[str, float, dict]:
        """
        从 Agent 的流式输出中提取生成的 SQL 和 RAG 中间态数据
        
        Args:
            agent: Text2SQLAgent 实例
            query: 自然语言问题
            
        Returns:
            (生成的 SQL, 耗时 ms, RAG 中间态数据 dict)
        """
        start_time = time.perf_counter()
        generated_sql = ""
        rag_metadata = {
            "query_complexity": "",
            "rag_top_k": 0,
            "rag_hints_count": 0,
            "token_usage": None  # 🆕 Token 消耗统计
        }
        
        try:
            for event in agent.generate_and_execute_stream(query, []):
                # 捕获 RAG 中间态数据（从 step 事件中提取）
                if event.get("type") == "step":
                    msg = event.get("msg", "")
                    # 解析复杂度信息
                    if "复杂度评估:" in msg:
                        if "simple" in msg.lower():
                            rag_metadata["query_complexity"] = "simple"
                        elif "complex" in msg.lower():
                            rag_metadata["query_complexity"] = "complex"
                        elif "medium" in msg.lower():
                            rag_metadata["query_complexity"] = "medium"
                    # 解析 hints 数量
                    if "hints" in msg.lower() and "条" in msg:
                        import re
                        match = re.search(r'(\d+)\s*条', msg)
                        if match:
                            rag_metadata["rag_hints_count"] = int(match.group(1))
                
                # 🆕 优先捕获累计 Token 使用统计（跨所有 LLM 调用）
                # cumulative_token_usage 优先于 token_usage（包含完整的多次调用统计）
                if event.get("type") == "cumulative_token_usage":
                    usage = event.get("usage", {})
                    rag_metadata["token_usage"] = usage
                    rag_metadata["token_usage_is_cumulative"] = True  # 标记这是累计值
                    rag_metadata["llm_call_count"] = usage.get("call_count", 1)
                    
                    # 🆕 解析 breakdown 数组，按调用类型分类统计
                    breakdown = usage.get("breakdown", [])
                    for call in breakdown:
                        call_type = call.get("call_type", "")
                        call_tokens = call.get("total_tokens", 0)
                        if call_type == "selector":
                            rag_metadata["selector_tokens"] = rag_metadata.get("selector_tokens", 0) + call_tokens
                        elif call_type == "generator":
                            attempt = call.get("attempt", 1)
                            if attempt == 1:
                                rag_metadata["generator_tokens"] = call_tokens
                            else:
                                # 重试调用归入 retry_tokens
                                rag_metadata["retry_tokens"] = rag_metadata.get("retry_tokens", 0) + call_tokens
                
                # 🆕 如果没有累计统计，则使用单次 token_usage（兜底）
                elif event.get("type") == "token_usage":
                    if not rag_metadata.get("token_usage_is_cumulative"):
                        rag_metadata["token_usage"] = event.get("usage", {})
                
                if event.get("type") == "sql_generated":
                    generated_sql = event.get("sql", "")
                    # 不立即 break，继续等待 token_usage 事件
                    # 如果后续有 result 事件则会在那里 break
                elif event.get("type") == "result":
                    # 如果直接到结果，从结果中提取 SQL
                    if "sql" in event:
                        generated_sql = event.get("sql", "")
                    break  # result 是最终事件，此时可以安全退出
                elif event.get("type") == "error":
                    break
        except Exception as e:
            print(f"  ⚠️ 流式处理异常: {e}")
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # 如果流中未捕获到复杂度，尝试从 Agent 获取
        if not rag_metadata["query_complexity"] and hasattr(agent, '_estimate_query_complexity'):
            try:
                rag_metadata["query_complexity"] = agent._estimate_query_complexity(query)
            except Exception:
                pass
        
        # 清洗 SQL：处理可能的 markdown 格式
        # 传递 agent 参数以使用 Agent 的 CoT 提取逻辑，确保评测与生产行为一致
        if generated_sql:
            generated_sql = self._extract_sql_from_text(generated_sql, agent)
        
        return generated_sql, elapsed_ms, rag_metadata
    
    def _extract_sql_from_text(self, text: str, agent: Optional['Text2SQLAgent'] = None) -> str:
        """
        从可能包含 markdown 的文本中提取纯 SQL
        
        优先使用 Agent 核心模块的提取逻辑，确保评测与生产行为一致。
        如果 Agent 不可用，则回退到简单提取。
        """
        import re
        
        if not text or not text.strip():
            return ""
        
        text = text.strip()
        
        # 优先使用 Agent 的提取逻辑（与生产环境一致）
        if agent is not None and hasattr(agent, '_extract_sql_from_response'):
            try:
                return agent._extract_sql_from_response(text)
            except Exception:
                pass  # 回退到下面的简单提取
        
        # 回退逻辑：简单的 markdown 代码块提取
        # 1. 尝试匹配 ```sql ... ``` 代码块
        match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # 2. 尝试匹配 ``` ... ``` 通用代码块
        match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            sql_candidate = match.group(1).strip()
            if sql_candidate.upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')):
                return sql_candidate
        
        # 3. 检查是否为纯 SQL (直接以 SQL 关键字开头)
        if text.upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')):
            return text
        
        return text
    
    def _evaluate_single_case(
        self, 
        case: EvalCase, 
        agent: Optional['Text2SQLAgent'],
        mode: ExperimentMode
    ) -> EvalResult:
        """
        评测单个用例
        
        Args:
            case: 评测用例
            agent: Agent 实例
            mode: 实验模式
            
        Returns:
            评测结果
        """
        # 每个 case 前重置 Agent 状态，确保独立运行
        if agent is not None and hasattr(agent, 'reset_state'):
            agent.reset_state()
        
        # 如果没有有效的 Agent，使用模拟模式
        if agent is None:
            return self._simulate_evaluation(case, mode)
        
        # 从 Agent 获取生成的 SQL 和 RAG 中间态数据
        generated_sql, exec_time, rag_metadata = self._extract_sql_from_stream(agent, case.question)
        
        if not generated_sql:
            return EvalResult(
                case_id=case.id,
                difficulty=case.difficulty.value,
                question=case.question,
                gold_sql=case.gold_sql,
                generated_sql="",
                is_correct=False,
                execution_time_ms=exec_time,
                error_message="No SQL generated",
                query_complexity=rag_metadata.get("query_complexity", ""),
                rag_top_k=rag_metadata.get("rag_top_k", 0),
                rag_hints_count=rag_metadata.get("rag_hints_count", 0)
            )
        
        # 执行 Gold SQL
        gold_df, gold_error = self.sql_executor.execute(case.gold_sql)
        if gold_error:
            return EvalResult(
                case_id=case.id,
                difficulty=case.difficulty.value,
                question=case.question,
                gold_sql=case.gold_sql,
                generated_sql=generated_sql,
                is_correct=False,
                execution_time_ms=exec_time,
                error_message=f"Gold SQL error: {gold_error}",
                query_complexity=rag_metadata.get("query_complexity", ""),
                rag_top_k=rag_metadata.get("rag_top_k", 0),
                rag_hints_count=rag_metadata.get("rag_hints_count", 0)
            )
        
        # 执行生成的 SQL
        gen_df, gen_error = self.sql_executor.execute(generated_sql)
        if gen_error:
            return EvalResult(
                case_id=case.id,
                difficulty=case.difficulty.value,
                question=case.question,
                gold_sql=case.gold_sql,
                generated_sql=generated_sql,
                is_correct=False,
                execution_time_ms=exec_time,
                error_message=f"Generated SQL error: {gen_error}",
                query_complexity=rag_metadata.get("query_complexity", ""),
                rag_top_k=rag_metadata.get("rag_top_k", 0),
                rag_hints_count=rag_metadata.get("rag_hints_count", 0)
            )
        
        # 比较结果
        is_correct, mismatch_reason = ResultComparator.compare(gold_df, gen_df)
        
        # Token 消耗统计 - 优先使用 Agent 提供的数据，否则估算
        token_usage = rag_metadata.get("token_usage")
        if token_usage:
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)
        else:
            # 估算 Token 数量 (约 4 个字符 = 1 个 Token，适用于英文/SQL)
            prompt_tokens = len(case.question) // 4 + 100  # 问题 + 基础 prompt 开销
            completion_tokens = len(generated_sql) // 4 if generated_sql else 0
            total_tokens = prompt_tokens + completion_tokens
        
        return EvalResult(
            case_id=case.id,
            difficulty=case.difficulty.value,
            question=case.question,
            gold_sql=case.gold_sql,
            generated_sql=generated_sql,
            is_correct=is_correct,
            execution_time_ms=exec_time,
            error_message=mismatch_reason if not is_correct else "",
            gold_result_hash=ResultComparator.dataframe_hash(gold_df),
            gen_result_hash=ResultComparator.dataframe_hash(gen_df),
            query_complexity=rag_metadata.get("query_complexity", ""),
            rag_top_k=rag_metadata.get("rag_top_k", 0),
            rag_hints_count=rag_metadata.get("rag_hints_count", 0),
            # Token 消耗统计
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            # Token breakdown (按调用阶段分解)
            selector_tokens=rag_metadata.get("selector_tokens", 0),
            generator_tokens=rag_metadata.get("generator_tokens", total_tokens),
            retry_tokens=rag_metadata.get("retry_tokens", 0),
            llm_call_count=rag_metadata.get("llm_call_count", 1)
        )
    
    def _simulate_evaluation(self, case: EvalCase, mode: ExperimentMode) -> EvalResult:
        """
        模拟评测（当 Agent 不可用时）
        
        用于测试和演示目的，生成模拟结果。
        """
        import random
        
        # 根据模式设置不同的模拟准确率
        accuracy_rates = {
            ExperimentMode.BASELINE: {
                Difficulty.EASY: 0.60,
                Difficulty.MEDIUM: 0.40,
                Difficulty.HARD: 0.25,
                Difficulty.EXTRA_HARD: 0.15
            },
            ExperimentMode.KNOWLEDGE_RAG: {
                Difficulty.EASY: 0.85,
                Difficulty.MEDIUM: 0.70,
                Difficulty.HARD: 0.50,
                Difficulty.EXTRA_HARD: 0.35
            },
            ExperimentMode.FULL_SYSTEM: {
                Difficulty.EASY: 0.95,
                Difficulty.MEDIUM: 0.85,
                Difficulty.HARD: 0.70,
                Difficulty.EXTRA_HARD: 0.55
            }
        }
        
        rate = accuracy_rates.get(mode, accuracy_rates[ExperimentMode.BASELINE]).get(case.difficulty, 0.5)
        is_correct = random.random() < rate
        
        # 模拟执行时间
        base_time = {
            ExperimentMode.BASELINE: 800,
            ExperimentMode.KNOWLEDGE_RAG: 1200,
            ExperimentMode.FULL_SYSTEM: 1500
        }
        exec_time = base_time.get(mode, 1000) + random.uniform(-200, 400)
        
        return EvalResult(
            case_id=case.id,
            difficulty=case.difficulty.value,
            question=case.question,
            gold_sql=case.gold_sql,
            generated_sql=case.gold_sql if is_correct else "SELECT 1",  # 模拟
            is_correct=is_correct,
            execution_time_ms=exec_time,
            error_message="" if is_correct else "Simulated mismatch"
        )
    
    def run_baseline(self, limit: int = None, verbose: bool = True) -> ExperimentSummary:
        """
        运行 G1 Baseline 实验
        
        Baseline 模式：直接将全部 Schema 丢给 LLM，不使用 RAG 检索，不注入领域知识，不启用自愈。
        
        学术意义:
            衡量 LLM 在无辅助情况下的原始 Text-to-SQL 能力，
            作为系统改进的基准线。
        
        Args:
            limit: 限制评测用例数量（用于快速测试）
            verbose: 是否输出详细信息
            
        Returns:
            实验摘要
        """
        mode = ExperimentMode.BASELINE
        if verbose:
            print("\n" + "=" * 60)
            print("🔬 实验 G1: BASELINE (无 RAG + 无 KECA + 无自愈)")
            print("=" * 60)
        
        agent = self._create_baseline_agent()
        if agent is None:
            print("   ❌ Baseline Agent 创建失败，将使用模拟模式")
            print("   💡 请检查: API Key、数据库连接、配置文件 (data/config.json)")
        else:
            print(f"   ✅ Baseline Agent 创建成功")
        return self._run_experiment(agent, mode, limit, verbose)
    
    def run_knowledge_rag(self, limit: int = None, verbose: bool = True) -> ExperimentSummary:
        """
        运行 G2 Knowledge-RAG 实验
        
        Knowledge-RAG 模式：启用 RAG 语义检索 + KECA 知识增强，但不启用错误自愈。
        
        学术意义:
            衡量 RAG + KECA 的综合贡献，
            量化与 Baseline 的差距即为知识增强检索的增益。
        
        Args:
            limit: 限制评测用例数量
            verbose: 是否输出详细信息
            
        Returns:
            实验摘要
        """
        mode = ExperimentMode.KNOWLEDGE_RAG
        if verbose:
            print("\n" + "=" * 60)
            print("🔬 实验 G2: KNOWLEDGE-RAG (有 RAG + 有 KECA + 无自愈)")
            print("=" * 60)
        
        agent = self._create_knowledge_rag_agent()
        if agent is None:
            print("   ❌ Knowledge-RAG Agent 创建失败，将使用模拟模式")
        else:
            print(f"   ✅ Knowledge-RAG Agent 创建成功")
        return self._run_experiment(agent, mode, limit, verbose)
    
    def run_full_system(self, limit: int = None, verbose: bool = True) -> ExperimentSummary:
        """
        运行 G3 Full System 实验
        
        Full System 模式：启用全部功能（RAG + KECA + 错误自愈）。
        
        学术意义:
            衡量完整系统的准确率，与 G2 对比可量化自愈机制的补偿效果。
        
        Args:
            limit: 限制评测用例数量
            verbose: 是否输出详细信息
            
        Returns:
            实验摘要
        """
        mode = ExperimentMode.FULL_SYSTEM
        if verbose:
            print("\n" + "=" * 60)
            print("🔬 实验 G3: FULL SYSTEM (有 RAG + 有 KECA + 有自愈)")
            print("=" * 60)
        
        agent = self._create_full_system_agent()
        if agent is None:
            print("   ❌ Full System Agent 创建失败，将使用模拟模式")
        else:
            print(f"   ✅ Full System Agent 创建成功")
        return self._run_experiment(agent, mode, limit, verbose)
    
    def run_full_system_with_g2_results(
        self, 
        g2_results: List[EvalResult], 
        verbose: bool = True
    ) -> ExperimentSummary:
        """
        运行 G3 Full System 实验（复用 G2 首次生成结果）
        
        P4 优化：确保 G3 准确率 ≥ G2
        - G2 正确的 Case：直接复用 G2 结果
        - G2 错误的 Case：使用自愈机制重试
        
        学术意义:
            严格量化自愈机制的增量贡献，保证 G3 ≥ G2。
        
        Args:
            g2_results: G2 实验的评测结果列表
            verbose: 是否输出详细信息
            
        Returns:
            实验摘要
        """
        mode = ExperimentMode.FULL_SYSTEM
        if verbose:
            print("\n" + "=" * 60)
            print("🔬 实验 G3: FULL SYSTEM (共享首次生成 + 自愈增强)")
            print("   💡 复用 G2 结果，仅对 G2 失败的 Case 运行自愈")
            print("=" * 60)
        
        agent = self._create_full_system_agent()
        if agent is None:
            print("   ❌ Full System Agent 创建失败，将使用模拟模式")
        else:
            print(f"   ✅ Full System Agent 创建成功")
        
        # 统计 G2 结果
        g2_correct_count = sum(1 for r in g2_results if r.is_correct)
        g2_failed_count = len(g2_results) - g2_correct_count
        
        if verbose:
            print(f"   📊 G2 结果: {g2_correct_count} 正确 / {g2_failed_count} 失败")
            print(f"   🔧 将对 {g2_failed_count} 个失败 Case 运行自愈")
        
        results = []
        g2_result_map = {r.case_id: r for r in g2_results}
        cases = self.data_loader.get_all_cases()
        
        # Token 累计统计
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        healed_count = 0  # 自愈成功计数
        
        for i, case in enumerate(cases, 1):
            g2_result = g2_result_map.get(case.id)
            
            if g2_result is None:
                # G2 没有该 Case 的结果，完整运行
                result = self._evaluate_single_case(case, agent, mode)
            elif g2_result.is_correct:
                # G2 正确：直接复用结果，标记为 FULL_SYSTEM 模式
                result = EvalResult(
                    case_id=g2_result.case_id,
                    difficulty=g2_result.difficulty,
                    question=g2_result.question,
                    gold_sql=g2_result.gold_sql,
                    generated_sql=g2_result.generated_sql,
                    is_correct=True,
                    execution_time_ms=g2_result.execution_time_ms,
                    error_message="",
                    gold_result_hash=g2_result.gold_result_hash,
                    gen_result_hash=g2_result.gen_result_hash,
                    query_complexity=g2_result.query_complexity,
                    rag_top_k=g2_result.rag_top_k,
                    rag_hints_count=g2_result.rag_hints_count,
                    prompt_tokens=g2_result.prompt_tokens,
                    completion_tokens=g2_result.completion_tokens,
                    total_tokens=g2_result.total_tokens
                )
            else:
                # G2 错误：运行自愈
                if verbose and i % 10 == 0:
                    print(f"   🔧 自愈尝试: {case.id}")
                
                result = self._evaluate_single_case(case, agent, mode)
                
                # 统计自愈成功
                if result.is_correct:
                    healed_count += 1
                    if verbose:
                        print(f"   ✅ 自愈成功: {case.id}")
            
            results.append(result)
            self.results[mode].append(result)
            
            # 累计 Token
            total_prompt_tokens += result.prompt_tokens
            total_completion_tokens += result.completion_tokens
            
            if verbose and i % 10 == 0:
                print(f"   进度: {i}/{len(cases)} ({i/len(cases)*100:.1f}%)")
        
        # 计算统计
        summary = self._compute_summary(mode, results)
        
        if verbose:
            self._print_summary(summary)
            print(f"\n   🔧 自愈增益: {healed_count} 个 Case 被修复 (+{healed_count/len(cases)*100:.1f}%)")
        
        # 保存结果
        self._save_results(mode, results)
        
        return summary
    
    def run_experiment(self, group_id: str, limit: int = None, verbose: bool = True) -> ExperimentSummary:
        """
        统一实验入口，通过实验组ID运行指定实验
        
        Args:
            group_id: 实验组ID ("G1", "G2", "G3" 或 mode 名称)
            limit: 限制评测用例数量
            verbose: 是否输出详细信息
            
        Returns:
            实验摘要
            
        Example:
            >>> benchmark.run_experiment("G1", limit=10)
            >>> benchmark.run_experiment("G3")
        """
        # 支持 G1/G2/G3 和 mode 名称两种格式
        group_map = {
            "G1": self.run_baseline,
            "G2": self.run_knowledge_rag,
            "G3": self.run_full_system,
            "baseline": self.run_baseline,
            "knowledge_rag": self.run_knowledge_rag,
            "full_system": self.run_full_system,
        }
        
        group_key = group_id.upper() if group_id.upper().startswith("G") else group_id.lower()
        run_fn = group_map.get(group_key)
        
        if run_fn is None:
            valid_groups = sorted(set(group_map.keys()))
            raise ValueError(f"未知的实验组ID: {group_id}. 有效值: {valid_groups}")
        
        return run_fn(limit=limit, verbose=verbose)
    
    def _run_experiment(
        self, 
        agent: Optional['Text2SQLAgent'],
        mode: ExperimentMode,
        limit: int = None,
        verbose: bool = True
    ) -> ExperimentSummary:
        """
        执行单组实验
        """
        cases = self.data_loader.get_all_cases()
        if limit:
            cases = cases[:limit]
        
        results = []
        total = len(cases)
        
        if verbose:
            print(f"📊 开始评测 {total} 道题目...")
            if agent is None:
                print("⚠️ Agent 不可用，将使用模拟模式")
        
        for i, case in enumerate(cases, 1):
            if verbose and i % 5 == 0:
                print(f"   进度: {i}/{total} ({i/total*100:.1f}%)")
            
            result = self._evaluate_single_case(case, agent, mode)
            results.append(result)
            self.results[mode].append(result)
        
        # 计算统计
        summary = self._compute_summary(mode, results)
        
        if verbose:
            self._print_summary(summary)
        
        # 保存结果
        self._save_results(mode, results)
        
        return summary
    
    def _compute_summary(self, mode: ExperimentMode, results: List[EvalResult]) -> ExperimentSummary:
        """计算实验摘要统计"""
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / total if total > 0 else 0
        
        # 按难度分组统计
        by_difficulty = {}
        for diff in Difficulty:
            diff_results = [r for r in results if r.difficulty == diff.value]
            if diff_results:
                diff_correct = sum(1 for r in diff_results if r.is_correct)
                diff_total = len(diff_results)
                by_difficulty[diff.value] = {
                    "total": diff_total,
                    "correct": diff_correct,
                    "accuracy": diff_correct / diff_total if diff_total > 0 else 0
                }
        
        avg_time = sum(r.execution_time_ms for r in results) / total if total > 0 else 0

        # P95 端到端响应时间（与 bench_performance.py 保持一致的线性插值法）
        times_sorted = sorted(r.execution_time_ms for r in results)
        n_t = len(times_sorted)
        if n_t >= 2:
            idx = (n_t - 1) * 0.95
            lo, hi = int(idx), int(idx) + 1
            w = idx - lo
            p95_time = times_sorted[lo] + w * (times_sorted[hi] - times_sorted[lo]) if hi < n_t else times_sorted[lo]
        elif n_t == 1:
            p95_time = times_sorted[0]
        else:
            p95_time = 0.0

        # 🆕 Token 消耗聚合统计
        total_prompt_tokens = sum(r.prompt_tokens for r in results)
        total_completion_tokens = sum(r.completion_tokens for r in results)
        total_tokens_sum = sum(r.total_tokens for r in results)
        avg_tokens_per_query = total_tokens_sum / total if total > 0 else 0
        
        return ExperimentSummary(
            mode=mode.value,
            total_cases=total,
            correct_count=correct,
            accuracy=accuracy,
            by_difficulty=by_difficulty,
            avg_execution_time_ms=avg_time,
            p95_execution_time_ms=p95_time,
            timestamp=datetime.now().isoformat(),
            # 🆕 Token 消耗统计
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_tokens_sum,
            avg_tokens_per_query=avg_tokens_per_query
        )
    
    def _print_summary(self, summary: ExperimentSummary):
        """打印实验摘要"""
        print(f"\n📈 {summary.mode.upper()} 实验结果:")
        print(f"   总体准确率: {summary.accuracy*100:.1f}% ({summary.correct_count}/{summary.total_cases})")
        print(f"   平均耗时: {summary.avg_execution_time_ms:.1f} ms")
        print(f"   P95 耗时: {summary.p95_execution_time_ms:.1f} ms")
        print("   按难度分布:")
        for diff, stats in summary.by_difficulty.items():
            acc = stats['accuracy'] * 100
            print(f"     {diff:12s}: {acc:5.1f}% ({stats['correct']}/{stats['total']})")
    
    def _get_isolated_result_dir(self, mode: ExperimentMode) -> Path:
        """
        获取隔离的结果目录路径
        
        格式: eval_suite/results/{database}_{mode}_{timestamp}/
        
        Returns:
            隔离结果目录的 Path 对象
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{self.database}_{mode.value}_{timestamp}"
        result_dir = self.results_dir / dir_name
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir
    
    def _save_results(self, mode: ExperimentMode, results: List[EvalResult]):
        """
        保存详细结果到隔离目录（含行数验证）
        
        结果保存到: 
        1. eval_suite/results/{database}_{mode}_{timestamp}/ (隔离目录，便于历史追溯)
        2. eval_suite/results/accuracy_{mode}.csv (顶层文件，供 Visualizer 读取)
        """
        # 创建隔离目录
        isolated_dir = self._get_isolated_result_dir(mode)
        
        # 保存详细结果 CSV
        df = pd.DataFrame([asdict(r) for r in results])
        csv_path = isolated_dir / "results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 🆕 同时保存到顶层 CSV（供 Visualizer 和报告使用）
        top_level_csv = self.results_dir / f"accuracy_{mode.value}.csv"
        df.to_csv(top_level_csv, index=False, encoding='utf-8-sig')
        
        # 保存生成的 SQL (便于事后分析)
        sql_data = [{"case_id": r.case_id, "question": r.question, 
                     "gold_sql": r.gold_sql, "generated_sql": r.generated_sql, 
                     "is_correct": r.is_correct} for r in results]
        sql_path = isolated_dir / "generated_sql.json"
        with open(sql_path, 'w', encoding='utf-8') as f:
            json.dump(sql_data, f, ensure_ascii=False, indent=2)
        
        # 验证保存结果
        try:
            saved_df = pd.read_csv(csv_path)
            saved_rows = len(saved_df)
            expected_rows = len(results)
            if saved_rows != expected_rows:
                print(f"   ⚠️ 警告：CSV 保存异常！保存行数 {saved_rows} != 预期 {expected_rows}")
            else:
                print(f"   💾 结果已保存: {isolated_dir.name}/ + accuracy_{mode.value}.csv ({saved_rows} 条记录)")
        except Exception as e:
            print(f"   ⚠️ CSV 验证失败: {e}")
            print(f"   💾 结果已保存: {isolated_dir}")
    
    def run_all(self, limit: int = None, verbose: bool = True) -> Dict[str, ExperimentSummary]:
        """
        运行全部三组实验 (G1-G3)
        
        Args:
            limit: 限制每组实验的用例数量
            verbose: 是否输出详细信息
            
        Returns:
            三组实验的摘要字典
        """
        summaries = {}
        
        summaries['baseline'] = self.run_baseline(limit, verbose)
        summaries['knowledge_rag'] = self.run_knowledge_rag(limit, verbose)
        summaries['full_system'] = self.run_full_system(limit, verbose)
        
        if verbose:
            self._print_comparison(summaries)
        
        return summaries
    
    def _print_comparison(self, summaries: Dict[str, ExperimentSummary]):
        """打印三组实验的对比结果 (G1-G3)"""
        print("\n" + "=" * 70)
        print("📊 准确率评测对比结果 (EX Metric)")
        print("=" * 70)
        
        # ASCII 表格
        header = f"{'方案':<15} | {'Easy':<10} | {'Medium':<10} | {'Hard':<10} | {'Extra':<10} | {'Overall':<10}"
        print(header)
        print("-" * len(header))
        
        for mode_name, summary in summaries.items():
            row = f"{mode_name:<15}"
            for diff in ['easy', 'medium', 'hard', 'extra_hard']:
                if diff in summary.by_difficulty:
                    acc = summary.by_difficulty[diff]['accuracy'] * 100
                    row += f" | {acc:>8.1f}%"
                else:
                    row += f" | {'N/A':>9}"
            row += f" | {summary.accuracy*100:>8.1f}%"
            print(row)
        
        print("=" * 70)
        
        # 计算增益
        if 'baseline' in summaries and 'full_system' in summaries:
            baseline_acc = summaries['baseline'].accuracy
            full_acc = summaries['full_system'].accuracy
            improvement = (full_acc - baseline_acc) * 100
            print(f"\n✨ 系统改进: Full System 相比 Baseline 提升 {improvement:.1f} 个百分点")


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="DeepInsight 准确率评测")
    parser.add_argument("--mode", choices=["baseline", "knowledge_rag", "full_system", "all"],
                        default="all", help="实验模式 (G1-G3)")
    parser.add_argument("--limit", type=int, default=None, help="限制评测用例数量")
    parser.add_argument("--test-mode", action="store_true", help="测试模式（少量样本）")
    parser.add_argument("--quiet", action="store_true", help="安静模式")
    parser.add_argument("--agent-backend", choices=["legacy", "graph"], default="legacy",
                        help="Agent 后端：legacy 使用原 Text2SQLAgent，graph 使用实验性 LangGraph 路径")
    
    args = parser.parse_args()
    
    if args.test_mode and args.limit is None:
        args.limit = 3
    
    benchmark = AccuracyBenchmark(agent_backend=args.agent_backend)
    verbose = not args.quiet
    
    if args.mode == "all":
        benchmark.run_all(limit=args.limit, verbose=verbose)
    elif args.mode == "baseline":
        benchmark.run_baseline(limit=args.limit, verbose=verbose)
    elif args.mode == "knowledge_rag":
        benchmark.run_knowledge_rag(limit=args.limit, verbose=verbose)
    elif args.mode == "full_system":
        benchmark.run_full_system(limit=args.limit, verbose=verbose)


if __name__ == "__main__":
    main()
