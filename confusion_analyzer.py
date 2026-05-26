from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

# V2 标签等价对集成
try:
    from confusion_matrix.label_config import LABEL_EQUIVALENCE_PAIRS
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False
    LABEL_EQUIVALENCE_PAIRS = {}


@dataclass
class ConfusionAnalyzer:
    """最小可用的混淆分析器。

    读取 `label_confusion_analysis.xlsx`（或同结构文件），按「真实标签==target」取 top-k 的「错误预测为」。

    期望列：
    - 真实标签
    - 错误预测为
    - 错误次数（可选，用于排序）
    - 混淆率（可选，用于排序）
    """

    path: str

    def __post_init__(self) -> None:
        self.df = pd.read_excel(self.path) if self.path.endswith(".xlsx") else pd.read_csv(self.path)
        required = {"真实标签", "错误预测为"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"confusion_file 缺少列: {sorted(missing)}，当前列: {self.df.columns.tolist()}")

    def get_confused_labels(self, target_label: str, top_k: Optional[int] = None) -> List[str]:
        """
        获取与目标标签混淆的标签列表，包括Excel中的混淆对和V2等价对。
        
        Args:
            target_label: 目标标签名称
            top_k: 返回前k个混淆标签，如果为None则返回全部混淆标签（默认）
        
        Returns:
            混淆标签列表
        """
        sub = self.df[self.df["真实标签"] == target_label].copy()
        
        # 从Excel中获取混淆标签
        excel_labels = []
        if len(sub) > 0:
            sort_col: Optional[str] = None
            for candidate in ("混淆率", "错误次数"):
                if candidate in sub.columns:
                    sort_col = candidate
                    break

            if sort_col is not None:
                # 确保排序列不包含 NaN 值，并转换为数值类型
                sub[sort_col] = pd.to_numeric(sub[sort_col], errors='coerce').fillna(0)
                sub = sub.sort_values(sort_col, ascending=False)

            excel_labels = sub["错误预测为"].astype(str).map(normalize_label).tolist()
        
        # 从V2等价对中获取标签
        equivalence_labels = []
        if V2_AVAILABLE and target_label in LABEL_EQUIVALENCE_PAIRS:
            equivalence_labels = LABEL_EQUIVALENCE_PAIRS[target_label]
        
        # 合并：Excel优先，然后加入等价对（去重）
        all_labels = excel_labels + equivalence_labels
        uniq: List[str] = []
        seen = set()
        for lbl in all_labels:
            if lbl and lbl not in seen and lbl != target_label:
                uniq.append(lbl)
                seen.add(lbl)
            # 如果指定了top_k，则限制数量；否则获取全部
            if top_k is not None and len(uniq) >= top_k:
                break
        
        return uniq if uniq else excel_labels if excel_labels else equivalence_labels


def normalize_label(value: str) -> str:
    # 清理一些常见脏数据（空、nan等）
    v = str(value).strip()
    if v.lower() in {"nan", "none", ""}:
        return ""
    return v
