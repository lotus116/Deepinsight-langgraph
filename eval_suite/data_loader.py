"""
评测数据加载器
=============

功能：
1. 加载 evaluation_set.json 评测数据集
2. 按 Spider 标准难度分组 (Easy/Medium/Hard/Extra-Hard)
3. 提供迭代器接口用于批量评测
4. 统计各难度级别的题目数量

学术背景：
    Spider 是 Text-to-SQL 领域最权威的跨域评测基准，
    本模块遵循其难度划分标准，便于与其他研究工作进行对比。
"""

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Iterator, Tuple
from enum import Enum
from pathlib import Path


class Difficulty(Enum):
    """
    Spider 标准难度等级
    
    划分依据 (根据 Spider 论文):
    - Easy: 单表简单查询，无嵌套
    - Medium: 包含 GROUP BY, HAVING, 或嵌套条件
    - Hard: 包含 JOIN, 子查询
    - Extra-Hard: 复杂嵌套、UNION、EXCEPT、窗口函数
    """
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTRA_HARD = "extra_hard"


@dataclass
class EvalCase:
    """
    评测用例数据结构
    
    Attributes:
        id: 唯一标识符 (如 q001)
        difficulty: Spider 标准难度等级
        question: 自然语言问题
        gold_sql: 标准答案 SQL
        expected_result_type: 预期结果类型 (single_value/table/empty)
        tags: 特征标签列表，用于细粒度分析
        description: 可选的题目描述/注释
    """
    id: str
    difficulty: Difficulty
    question: str
    gold_sql: str
    expected_result_type: str = "table"
    tags: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        # 如果 difficulty 是字符串，转换为枚举
        if isinstance(self.difficulty, str):
            self.difficulty = Difficulty(self.difficulty.lower())


class EvalDataLoader:
    """
    评测数据加载器
    
    负责加载、解析和组织评测数据集，提供按难度分组的迭代器接口。
    
    Example:
        >>> loader = EvalDataLoader()
        >>> for case in loader.iter_by_difficulty(Difficulty.EASY):
        ...     print(case.question)
        >>> stats = loader.get_stats()
        >>> print(stats)
    """
    
    def __init__(self, data_path: str = None, database: str = "northwind"):
        """
        初始化数据加载器
        
        Args:
            data_path: evaluation_set.json 的路径，默认根据 database 参数自动确定
            database: 数据库名称 ("northwind" 或 "adventureworks")
        """
        if data_path is None:
            # 默认路径：相对于本文件的同级目录
            current_dir = Path(__file__).parent
            if database.lower() == "adventureworks":
                data_path = current_dir / "evaluation_set_adventureworks.json"
            else:
                data_path = current_dir / "evaluation_set.json"
        
        self.data_path = Path(data_path)
        self.database = database
        self.cases: List[EvalCase] = []
        self._grouped_cases: Dict[Difficulty, List[EvalCase]] = {
            d: [] for d in Difficulty
        }
        
        self._load_data()
    
    def _load_data(self):
        """加载并解析评测数据"""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"评测数据集不存在: {self.data_path}\n"
                f"请确保 evaluation_set.json 文件位于 eval_suite 目录下"
            )
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        for item in raw_data:
            case = EvalCase(
                id=item.get("id", "unknown"),
                difficulty=item.get("difficulty", "medium"),
                question=item.get("question", ""),
                gold_sql=item.get("gold_sql", ""),
                expected_result_type=item.get("expected_result_type", "table"),
                tags=item.get("tags", []),
                description=item.get("description", "")
            )
            self.cases.append(case)
            self._grouped_cases[case.difficulty].append(case)
        
        print(f"✅ 成功加载 {len(self.cases)} 道评测题目")
    
    def get_all_cases(self) -> List[EvalCase]:
        """获取所有评测用例"""
        return self.cases
    
    def get_cases_by_difficulty(self, difficulty: Difficulty) -> List[EvalCase]:
        """
        按难度获取评测用例
        
        Args:
            difficulty: Spider 难度等级
            
        Returns:
            该难度下的所有评测用例列表
        """
        return self._grouped_cases.get(difficulty, [])
    
    def iter_all(self) -> Iterator[EvalCase]:
        """迭代所有评测用例"""
        yield from self.cases
    
    def iter_by_difficulty(self, difficulty: Difficulty) -> Iterator[EvalCase]:
        """
        按难度迭代评测用例
        
        Args:
            difficulty: Spider 难度等级
            
        Yields:
            该难度下的评测用例
        """
        yield from self._grouped_cases.get(difficulty, [])
    
    def iter_with_difficulty(self) -> Iterator[Tuple[Difficulty, EvalCase]]:
        """
        迭代所有用例，同时返回难度标签
        
        Yields:
            (难度, 评测用例) 元组
        """
        for case in self.cases:
            yield case.difficulty, case
    
    def get_stats(self) -> Dict[str, any]:
        """
        获取数据集统计信息
        
        Returns:
            包含总数、各难度分布、标签分布的统计字典
            
        学术意义:
            统计信息用于论文中描述评测数据集的分布特征，
            确保评测的公平性和可重复性。
        """
        tag_counts = {}
        for case in self.cases:
            for tag in case.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return {
            "total": len(self.cases),
            "by_difficulty": {
                d.value: len(self._grouped_cases[d]) for d in Difficulty
            },
            "by_tags": tag_counts,
            "result_types": self._count_result_types()
        }
    
    def _count_result_types(self) -> Dict[str, int]:
        """统计结果类型分布"""
        counts = {}
        for case in self.cases:
            rt = case.expected_result_type
            counts[rt] = counts.get(rt, 0) + 1
        return counts
    
    def get_case_by_id(self, case_id: str) -> Optional[EvalCase]:
        """
        按 ID 获取特定评测用例
        
        Args:
            case_id: 用例 ID (如 "q001")
            
        Returns:
            匹配的评测用例，未找到返回 None
        """
        for case in self.cases:
            if case.id == case_id:
                return case
        return None
    
    def sample(self, n: int = 5, difficulty: Difficulty = None) -> List[EvalCase]:
        """
        随机采样评测用例
        
        Args:
            n: 采样数量
            difficulty: 可选，限制采样的难度级别
            
        Returns:
            采样的评测用例列表
        """
        import random
        
        pool = self._grouped_cases[difficulty] if difficulty else self.cases
        n = min(n, len(pool))
        return random.sample(pool, n)


def print_dataset_summary(loader: EvalDataLoader):
    """
    打印数据集摘要（用于终端展示）
    
    生成一个格式化的表格，展示数据集的分布情况。
    """
    stats = loader.get_stats()
    
    print("\n" + "=" * 50)
    print("📊 DeepInsight 评测数据集摘要")
    print("=" * 50)
    print(f"总题目数: {stats['total']}")
    print("-" * 50)
    print("按难度分布:")
    for diff, count in stats['by_difficulty'].items():
        percentage = (count / stats['total']) * 100 if stats['total'] > 0 else 0
        bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
        print(f"  {diff:12s} | {bar} | {count:3d} ({percentage:5.1f}%)")
    print("-" * 50)
    print("按结果类型分布:")
    for rt, count in stats['result_types'].items():
        print(f"  {rt:15s}: {count}")
    print("=" * 50 + "\n")


# ============================================================
# 模块测试入口
# ============================================================
if __name__ == "__main__":
    try:
        loader = EvalDataLoader()
        print_dataset_summary(loader)
        
        # 展示每个难度的示例题目
        for diff in Difficulty:
            cases = loader.get_cases_by_difficulty(diff)
            if cases:
                print(f"\n📝 {diff.value.upper()} 难度示例:")
                print(f"   问题: {cases[0].question}")
                print(f"   SQL:  {cases[0].gold_sql[:60]}...")
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        print("请先运行评测套件以生成评测数据集")
