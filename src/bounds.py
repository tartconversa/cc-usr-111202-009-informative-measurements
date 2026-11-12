"""可证明信息上界与剪枝辅助。

目标函数

    f(S) = 0.5 (log det P(S) - log det P0)

是单调子模函数（协方差正定下边际增益随集合增大而递减）。一般基数/预算
约束下贪心没有固定近似比保证，因此本模块不声称任何比率，而是构造若干
*合法上界* 并取最紧者（:func:`combined_upper_bound`）：

* :func:`relaxation_upper_bound` —— 预算松弛，按初始边际/成本比扫描，
  装不进项计入完整初始边际（子模性保证其为真实边际的上界）；
* :func:`cardinality_upper_bound` —— 预算基数限制，取最大的 K 个初始
  边际之和；
* :func:`partition_aware_upper_bound` —— 互斥/位置分区的逐区最大边际和
  （仅根空状态）。

合法性统一来自子模性：候选在初始松状态上的边际不小于它在任意后续更紧
状态上的真实边际。上界配合可行解给出可证明 gap，不使用分数插值（离散
子模函数不满足背包可分性，分数项会把上界压到最优值以下）。
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from .design_model import Candidate
from .information_state import ConstraintState, InformationState

_GAP_TOL = 1e-12


def relaxation_upper_bound(
    info: InformationState,
    constraints: ConstraintState,
    candidates: Sequence[Candidate],
) -> float:
    """预算松弛上界（nats），基于 ``info``/``constraints`` 的当前状态。

    ``candidates`` 为仍可考虑的候选（分支定界中即未处理后缀）。

    构造（子模边际上界，可证明合法；已对全部可行子集枚举校验）：

    1. 放宽除总预算外的全部硬约束，与当前状态冲突的候选不进入松弛；
    2. 零成本候选先实际计入（真实边际直接累加，不耗预算）；
    3. 正成本候选在 *同一初始工作状态* 上估一次边际 ``m_j``（子模性保证
       它是该候选在任意后续更紧状态上真实边际的上界），按效率比
       ``m_j/c_j`` 降序；
    4. 顺序扫描：装得下就实际加入（累加不超过 ``m_j`` 的真实边际），
       装不下则把完整初始边际 ``m_j`` 计入上界并继续扫描更便宜的候选。

    合法性：任意预算内可行方案 T 的每个元素，其在 T 链上的真实边际不超过
    它在初始松状态上的边际 ``m_j``；扫描对实际加入项累加真实边际、对未加入
    项累加 ``m_j``，故返回值支配所有可行方案目标值。不使用分数插值——
    离散子模函数不满足背包可分性，分数项会把上界压到最优值以下。这是上界
    而非近似比；一般约束下不声称任何固定近似比，只报告与可行解的 gap。
    """
    remaining_budget = constraints.problem.budget - constraints.cost
    if remaining_budget < -_GAP_TOL:
        return -math.inf  # 超预算状态：该子树无可行解

    # 组下限的最便宜追加成本只统计当前可加入候选；这是必要条件检查
    cheapest = constraints.cheapest_minima_cost(candidates)
    if cheapest is not None and cheapest > remaining_budget + _GAP_TOL:
        return -math.inf

    working = info.copy()
    positive: List[Candidate] = []
    for candidate in candidates:
        if constraints.block_reason(candidate) is not None:
            continue
        if candidate.cost <= _GAP_TOL:
            working.add(candidate)  # 零成本：实际计入，不耗预算
        else:
            positive.append(candidate)

    if not positive:
        return working.information_gain

    # 所有边际都在同一个初始工作状态上估计一次：它们是后续真实边际的上界
    scored = []
    for candidate in positive:
        marginal = working.marginal_gain(candidate)
        scored.append((marginal / candidate.cost, marginal, candidate))
    scored.sort(key=lambda item: (-item[0], item[2].key))  # 键次序确定性并列

    bound = working.information_gain
    spent = 0.0
    for _, marginal, candidate in scored:
        if spent + candidate.cost <= remaining_budget + _GAP_TOL:
            # 装得下：累加真实边际（<= 初始边际 m_j），并更新工作状态
            bound += working.add(candidate)
            spent += candidate.cost
        else:
            # 装不下：不实际加入，把完整初始边际上界 m_j 计入后继续
            bound += marginal
    return bound


def partition_aware_upper_bound(
    info: InformationState,
    constraints: ConstraintState,
    candidates: Sequence[Candidate],
) -> float:
    """基于互斥/位置分区的严格边际上界（仅根/空选择状态调用）。

    合法构造：在空状态上对每个候选估一次初始边际 ``m_j``（子模性保证它是
    任意后续真实边际的上界）。互斥标签把候选划分为互斥分区——任一可行方案
    在每个互斥分区内至多取一个候选，因此该分区对总增益的贡献上界是区内
    最大的 ``m_j``；无互斥标签的候选各自独立成区。位置上限同理，对同一
    位置的候选取最大的 ``position_upper_bound`` 个边际之和。两组分区上界
    取 min。全程不做贪心打包，故不存在“松弛选择挤掉最优候选”的非法问题。
    """
    problem = constraints.problem
    if constraints.selected:
        raise ValueError("partition_aware_upper_bound 仅支持空选择状态")

    marginals = {
        candidate.key: info.marginal_gain(candidate) for candidate in candidates
    }

    # 互斥分区：共享任一标签的候选归入同一分区（合并重叠标签）
    label_to_keys: dict = {}
    no_label: List[str] = []
    for candidate in candidates:
        if candidate.mutex:
            for label in candidate.mutex:
                label_to_keys.setdefault(label, set()).add(candidate.key)
        else:
            no_label.append(candidate.key)
    merged: List[set] = []
    for keys in label_to_keys.values():
        for group in merged:
            if group & keys:
                group |= keys
                break
        else:
            merged.append(set(keys))
    mutex_bound = sum(marginals[k] for k in no_label)
    for group in merged:
        mutex_bound += max(marginals[k] for k in group)

    # 位置分区：每位置至多 position_upper_bound 个，取该区最大的若干边际
    position_keys: dict = {}
    for candidate in candidates:
        position_keys.setdefault(candidate.position, []).append(candidate.key)
    position_bound = 0.0
    cap = problem.position_upper_bound
    for keys in position_keys.values():
        top = sorted((marginals[k] for k in keys), reverse=True)[:cap]
        position_bound += sum(top)

    return info.information_gain + min(mutex_bound, position_bound)


def cardinality_upper_bound(
    info: InformationState,
    constraints: ConstraintState,
    candidates: Sequence[Candidate],
) -> float:
    """预算基数上界：方案至多含 ``floor(剩余预算/最小成本)`` 个测量。

    取初始状态上最大的 K 个边际之和。子模性保证任意 K 元素方案的目标值不
    超过初始状态上最大的 K 个边际之和，故严格合法。
    """
    remaining_budget = constraints.problem.budget - constraints.cost
    eligible = [
        c for c in candidates if constraints.block_reason(c) is None
    ]
    if not eligible:
        return info.information_gain
    min_cost = min(c.cost for c in eligible)
    if min_cost <= _GAP_TOL:
        return math.inf  # 存在零成本候选时基数不受预算限制
    limit = int(remaining_budget // min_cost)
    marginals = sorted(
        (info.marginal_gain(c) for c in eligible), reverse=True
    )
    return info.information_gain + sum(marginals[:limit])


def combined_upper_bound(
    info: InformationState,
    constraints: ConstraintState,
    candidates: Sequence[Candidate],
) -> float:
    """多个合法上界取 min，收紧 gap 而不破坏可证明性。

    分支节点：预算松弛上界 + 基数上界（均对非空状态合法）。
    根节点：额外纳入互斥/位置分区上界。
    """
    first = relaxation_upper_bound(info, constraints, candidates)
    if not math.isfinite(first):
        return first
    bounds = [first, cardinality_upper_bound(info, constraints, candidates)]
    if not constraints.selected:
        bounds.append(
            partition_aware_upper_bound(info, constraints, candidates)
        )
    return min(bounds)


def root_upper_bound(problem, info_factory=None) -> float:
    """根节点上界：预算与分区约束松弛的组合上界，用于最终 gap 报告。"""
    from .information_state import InformationState

    info = info_factory() if info_factory else InformationState(problem)
    constraints = ConstraintState.empty(problem)
    return combined_upper_bound(info, constraints, problem.candidates)


def gap_values(feasible_gain: Optional[float], upper_bound: float) -> dict:
    """由可行解与上界计算绝对/相对 gap。"""
    if feasible_gain is None:
        return {
            "feasible_gain": None,
            "upper_bound": upper_bound if math.isfinite(upper_bound) else None,
            "absolute_gap": None,
            "relative_gap": None,
        }
    absolute = max(0.0, upper_bound - feasible_gain)
    denominator = max(abs(upper_bound), 1e-12)
    relative = absolute / denominator
    return {
        "feasible_gain": feasible_gain,
        "upper_bound": upper_bound,
        "absolute_gap": absolute,
        "relative_gap": relative,
    }
