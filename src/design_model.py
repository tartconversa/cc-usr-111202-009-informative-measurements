"""测量设计问题的数据模型、输入解析与契约校验。

本模块只描述 *先验与候选测量*，不接触任何观测响应：观测值尚未到达，
候选记录若携带响应类字段会被直接拒绝，选择过程也不读取任何数据相关量。
解析通过 :mod:`src.design_reader` 完成，该读取器保持只负责读配置。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import design_reader
from .linear_algebra import cholesky

# 观测值尚未到达：这些字段一旦出现在候选记录里即视为契约违规，
# 防止任何选择逻辑把未知测量结果当作选择特征。
FORBIDDEN_OBSERVATION_FIELDS = frozenset(
    {
        "observation",
        "observed_value",
        "response",
        "y",
        "outcome",
        "measurement_value",
        "result",
    }
)


class DesignContractError(ValueError):
    """输入不满足测量设计契约（维度、正定性、正噪声等）。"""


@dataclass(frozen=True)
class Candidate:
    """一个候选线性测量 y = a^T theta + e，e 独立、方差为 sigma^2。"""

    key: str
    row: Tuple[float, ...]
    noise_variance: float
    cost: float
    group: str
    position: str
    mutex: Tuple[str, ...]

    @property
    def scaled_row(self) -> Tuple[float, ...]:
        """u = a / sigma；信息贡献为 u u^T（与 a a^T / sigma^2 相同）。"""
        sigma = math.sqrt(self.noise_variance)
        return tuple(v / sigma for v in self.row)


@dataclass
class DesignProblem:
    """一次完整设计比较的全部静态输入。"""

    candidates: List[Candidate]
    prior_covariance: List[List[float]]
    budget: float
    group_min: Dict[str, int]
    group_max: Dict[str, int]
    position_upper_bound: int
    search_node_budget: int = 10_000
    noise_scale_sensitivity: Tuple[float, ...] = (1.0,)
    tie_break: str = "lexicographic_selected_keys"
    source_candidates: Optional[str] = None
    source_design: Optional[str] = None

    @property
    def dim(self) -> int:
        return len(self.prior_covariance)

    @property
    def n(self) -> int:
        return len(self.candidates)

    def candidate_by_key(self) -> Dict[str, Candidate]:
        return {c.key: c for c in self.candidates}

    def all_groups(self) -> List[str]:
        groups = set(self.group_min) | set(self.group_max)
        groups |= {c.group for c in self.candidates}
        return sorted(groups)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require_number(value: Any, what: str) -> float:
    if not _is_number(value):
        raise DesignContractError(f"{what} 必须是有限数值，得到 {value!r}")
    return float(value)


def _validate_prior(matrix: Any) -> List[List[float]]:
    if not isinstance(matrix, list) or not matrix or not all(
        isinstance(row, list) for row in matrix
    ):
        raise DesignContractError("prior_covariance 必须是非空二维数组")
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise DesignContractError("prior_covariance 必须是方阵")
    prior = [[_require_number(v, "prior_covariance 元素") for v in row] for row in matrix]
    for i in range(n):
        for j in range(i):
            if not math.isclose(prior[i][j], prior[j][i], rel_tol=1e-12, abs_tol=1e-12):
                raise DesignContractError("prior_covariance 必须对称")
    try:
        cholesky(prior)
    except Exception as exc:  # NotPositiveDefinite
        raise DesignContractError(
            f"先验协方差 Sigma0 必须正定，Cholesky 校验失败：{exc}"
        ) from exc
    return prior


def _parse_candidate(raw: Mapping[str, Any], dim: int, seen: set) -> Candidate:
    if not isinstance(raw, Mapping):
        raise DesignContractError("每个候选必须是对象")
    bad = FORBIDDEN_OBSERVATION_FIELDS.intersection(raw)
    if bad:
        raise DesignContractError(
            f"候选 {raw.get('key', '?')} 含观测类字段 {sorted(bad)}；"
            "观测值尚未到达，不允许以未知测量结果作为选择特征"
        )
    try:
        key = str(raw["key"])
    except KeyError:
        raise DesignContractError("候选缺少 key 字段") from None
    if key in seen:
        raise DesignContractError(f"候选 key 重复：{key}")
    seen.add(key)

    row = raw.get("row")
    if not isinstance(row, list) or len(row) != dim:
        raise DesignContractError(
            f"候选 {key} 的 row 必须是长度 {dim} 的数组"
        )
    parsed_row = tuple(_require_number(v, f"候选 {key} 的 row") for v in row)

    noise = _require_number(raw.get("noise_variance"), f"候选 {key} 的 noise_variance")
    if noise <= 0.0:
        raise DesignContractError(f"候选 {key} 的 noise_variance 必须为正")
    cost = _require_number(raw.get("cost"), f"候选 {key} 的 cost")
    if cost < 0.0:
        raise DesignContractError(f"候选 {key} 的 cost 不能为负")

    group = str(raw.get("group", "default"))
    position = str(raw.get("position", key))
    mutex_raw = raw.get("mutex", [])
    if not isinstance(mutex_raw, list) or not all(isinstance(m, str) for m in mutex_raw):
        raise DesignContractError(f"候选 {key} 的 mutex 必须是字符串数组")
    return Candidate(
        key=key,
        row=parsed_row,
        noise_variance=noise,
        cost=cost,
        group=group,
        position=position,
        mutex=tuple(mutex_raw),
    )


def _as_int(value: Any, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DesignContractError(f"{what} 必须是非负整数，得到 {value!r}")
    return value


def load_problem(
    candidates_path: str,
    design_path: Optional[str] = None,
) -> DesignProblem:
    """从 fixture 读取并校验问题。

    ``design_path`` 给定时以其为准；否则使用候选文件内嵌的设计字段
    （``candidates-large.json`` 即采用内嵌形式）。
    """
    raw_candidates = design_reader.load(candidates_path)
    if not isinstance(raw_candidates, Mapping) or "candidates" not in raw_candidates:
        raise DesignContractError("候选文件必须是含 candidates 数组的对象")

    if design_path is not None:
        design = design_reader.load(design_path)
        if not isinstance(design, Mapping):
            raise DesignContractError("设计文件必须是对象")
    else:
        design = raw_candidates

    prior = _validate_prior(design.get("prior_covariance"))
    dim = len(prior)

    seen: set = set()
    candidates = [
        _parse_candidate(raw, dim, seen) for raw in raw_candidates["candidates"]
    ]
    if not candidates:
        raise DesignContractError("候选集合为空，无法构建设计问题")
    # 内部统一按键字典序处理：保证重排输入时并列结果与固定键次序输出不变
    candidates.sort(key=lambda c: c.key)

    budget = _require_number(design.get("budget"), "budget")
    if budget < 0:
        raise DesignContractError("budget 不能为负")

    def _int_map(field_name: str, default: int) -> Dict[str, int]:
        raw = design.get(field_name, {})
        if not isinstance(raw, Mapping):
            raise DesignContractError(f"{field_name} 必须是字符串到整数的映射")
        return {str(g): _as_int(v, f"{field_name}[{g}]") for g, v in raw.items()}

    group_min = _int_map("group_min", 0)
    group_max = _int_map("group_max", 10**9)
    for group, lower in group_min.items():
        if lower > group_max.get(group, 10**9):
            raise DesignContractError(
                f"组 {group} 的 group_min={lower} 超过 group_max"
            )

    position_upper_bound = _as_int(
        design.get("position_upper_bound", 1), "position_upper_bound"
    )
    if position_upper_bound < 1:
        raise DesignContractError("position_upper_bound 至少为 1")

    node_budget = _as_int(
        design.get("search_node_budget", 10_000), "search_node_budget"
    )
    if node_budget < 1:
        raise DesignContractError("search_node_budget 至少为 1")

    scales_raw = design.get("noise_scale_sensitivity", [1.0])
    if not isinstance(scales_raw, list) or not scales_raw:
        raise DesignContractError("noise_scale_sensitivity 必须是非空数组")
    scales = tuple(
        _require_number(s, "noise_scale_sensitivity") for s in scales_raw
    )
    if any(s <= 0 for s in scales):
        raise DesignContractError("noise_scale_sensitivity 中的倍数必须为正")

    return DesignProblem(
        candidates=candidates,
        prior_covariance=prior,
        budget=budget,
        group_min=group_min,
        group_max=group_max,
        position_upper_bound=position_upper_bound,
        search_node_budget=node_budget,
        noise_scale_sensitivity=scales,
        tie_break=str(design.get("tie_break", "lexicographic_selected_keys")),
        source_candidates=candidates_path,
        source_design=design_path,
    )
