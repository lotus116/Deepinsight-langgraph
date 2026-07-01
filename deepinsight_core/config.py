"""Configuration helpers for service-oriented DeepInsight code.

The current project stores most settings in ``data/config.json`` and passes the
raw dictionary directly into Streamlit.  This module keeps
that dictionary compatible while giving the refactored services a typed,
central place to read defaults from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class DeepInsightSettings:
    api_key: str = ""
    api_base: str = "https://api.deepseek.com"
    model_name: str = "deepseek-reasoner"
    reasoner_model: Optional[str] = "deepseek-reasoner"
    use_reasoner_for_healing: bool = True

    db_uris: List[str] = field(default_factory=lambda: ["sqlite:///data/ecommerce.db"])
    model_path: str = "models/bge-small-ov"
    kb_paths: List[str] = field(default_factory=list)

    max_retries: int = 3
    max_candidates: int = 1
    log_file: str = "data/agent.log"

    chroma_path: str = "data/chroma"
    chroma_collection_prefix: str = "deepinsight"
    use_chroma_rag: bool = False

    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, config: Optional[Dict[str, Any]]) -> "DeepInsightSettings":
        config = dict(config or {})
        return cls(
            api_key=config.get("api_key", ""),
            api_base=config.get("api_base", config.get("base_url", "https://api.deepseek.com")),
            model_name=config.get("model_name", "deepseek-reasoner"),
            reasoner_model=config.get("reasoner_model", "deepseek-reasoner"),
            use_reasoner_for_healing=bool(config.get("use_reasoner_for_healing", True)),
            db_uris=list(config.get("db_uris") or ([config["db_path"]] if config.get("db_path") else [])),
            model_path=config.get("model_path", "models/bge-small-ov"),
            kb_paths=list(config.get("kb_paths_list") or config.get("kb_paths") or []),
            max_retries=int(config.get("max_retries", 3)),
            max_candidates=int(config.get("max_candidates", 1)),
            log_file=config.get("log_file", "data/agent.log"),
            chroma_path=config.get("chroma_path", "data/chroma"),
            chroma_collection_prefix=config.get("chroma_collection_prefix", "deepinsight"),
            use_chroma_rag=bool(config.get("use_chroma_rag", False)),
            raw=MappingProxyType(config),
        )

    @property
    def first_db_uri(self) -> str:
        return self.db_uris[0] if self.db_uris else ""

    def ensure_storage_dirs(self) -> None:
        for path in (self.log_file, self.chroma_path):
            target = Path(path)
            directory = target if target.suffix == "" else target.parent
            directory.mkdir(parents=True, exist_ok=True)
