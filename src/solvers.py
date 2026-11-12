"""两类求解器与“未选候选”解释。

* :func:`exact_solve` —— 小规模精确分支定界。固定候选键次序做包含/排除
  深度优先搜索，预算松弛上界剪枝；受 ``search_node_budget`` 限制停止时，
  返回当前可行解、可证明上界与 gap，状态标为 ``budget_stopped``。
* :func:`incremental_solve` —— 可扩展增量搜索。先做仅约束的可行性骨架
  满足组下限，再以懒评估贪心按边际/成本比逐次加入；每一步同时报告
  边际信息收益与效率比。一般约束下不声称固定近似比，结束时给上界与 gap。
* :func:`explain_exclusion` —— 对任一未选候选区分：
  约束不可加入（硬约束在最终方案上直接冲突）与重新优化后的全局机会
  损失（强制包含该候选再解一次，比较最优值）；后者不能用当下边际替代。

并列一律按候选键的字典序打破，保证重排输入时同一方案的固定键次序输出。
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .bounds import combined_upper_bound, gap_values
from .design_model import Candidate, DesignProblem
from .information_state import ConstraintState, InformationState

_TOL = 1e-12


@dataclass
class SelectionStep:
    order: int
    key: str
    marginal_gain: float
    gain_per_cost: float
    cost: float
    cumulative_gain: float
    cumulative_cost: float
    phase: str  # skeleton | greedy


@dataclass
class SolveResult:
    method: str
    status: str  # optimal | budget_stopped | infeasible
    selected: List[str]
    cost: float
    information_gain: float
    upper_bound: Optional[float]
    absolute_gap: Optional[float]
    relative_gap: Optional[float]
    nodes_visited: int
    node_budget: int
    steps: List[SelectionStep] = field(default_factory=list)
    infeasibility_reason: Optional[str] = None
    note: Optional[str] = None

    @property
    def is_feasible(self) -> bool:
        return self.status != "infeasible"


# --------------------------------------------------------------------------- #
# 仅约束的可行性深度优先搜索（不计算信息，速度快，用于骨架与不可行诊断）
# --------------------------------------------------------------------------- #
def _feasibility_dfs(
    problem: DesignProblem,
    node_budget: int,
    force_include: Optional[set] = None,
) -> Tuple[Optional[ConstraintState], int, Optional[str]]:
    candidates = problem.candidates
    force_include = force_include or set()
    counter = {"nodes": 0}

    def recurse(
        idx: int, state: ConstraintState
    ) -> Optional[ConstraintState]:
        counter["nodes"] += 1
        if counter["nodes"] > node_budget:
            return None
        # 剩余候选能否补齐组下限
        remaining_groups = [c.group for c in candidates[idx:]]
        if not state.minima_achievable(remaining_groups):
            return None
        # 注意：不能在此用“组内最便宜成本之和”剪预算——互斥/位置冲突会使
        # 这些便宜候选实际不可用，该估计不是合法下界，提前剪枝会误判无解。

        if idx == len(candidates):
            if state.minima_satisfied():
                return state
            return None

        candidate = candidates[idx]
        # 强制包含的候选不能走排除分支
        must_take = candidate.key in force_include

        if state.block_reason(candidate) is None:
            taken = state.copy()
            taken.add(candidate)
            found = recurse(idx + 1, taken)
            if found is not None:
                return found
        if not must_take:
            found = recurse(idx + 1, state.copy())
            if found is not None:
                return found
        return None

    found = recurse(0, ConstraintState.empty(problem))
    if found is not None:
        return found, counter["nodes"], None
    if counter["nodes"] > node_budget:
        return None, counter["nodes"], "可行性搜索在节点预算内未完成，无法判定可行性"
    return None, counter["nodes"], _diagnose_infeasibility(problem)


def _diagnose_infeasibility(problem: DesignProblem) -> str:
    """给出可读的不可行原因（检查各类硬约束的最小需求）。"""
    reasons: List[str] = []
    # 组下限自身是否可能
    group_pool: Dict[str, List[Candidate]] = {}
    for candidate in problem.candidates:
        group_pool.setdefault(candidate.group, []).append(candidate)
    for group, lower in problem.group_min.items():
        pool = group_pool.get(group, [])
        cap = problem.group_max.get(group, 10**9)
        if len(pool) < lower:
            reasons.append(
                f"组 {group} 要求至少 {lower} 个，但只有 {len(pool)} 个候选"
            )
        if lower > cap:
            reasons.append(f"组 {group} 下限 {lower} 超过上限 {cap}")
        cheapest = sum(sorted(c.cost for c in pool)[:lower]) if lower else 0.0
        if cheapest > problem.budget + _TOL:
            reasons.append(
                f"仅满足组 {group} 下限就至少需要成本 {cheapest:g}，超过预算 {problem.budget:g}"
            )
    # 互斥/位置与下限的全局冲突
    min_total = sum(problem.group_min.values())
    cheapest_all = sum(sorted(c.cost for c in problem.candidates)[:min_total])
    if cheapest_all > problem.budget + _TOL:
        reasons.append(
            f"满足全部组下限至少需要成本 {cheapest_all:g}，超过预算 {problem.budget:g}"
        )
    if not reasons:
        reasons.append("互斥标签、位置重复上限与组下限/预算组合后不存在可行方案")
    return "；".join(reasons)


# --------------------------------------------------------------------------- #
# 精确分支定界
# --------------------------------------------------------------------------- #
def exact_solve(
    problem: DesignProblem,
    force_include: Optional[set] = None,
    node_budget: Optional[int] = None,
) -> SolveResult:
    """小规模精确求解；节点预算耗尽时返回当前可行解、上界与 gap。"""
    force_include = set(force_include or ())
    unknown = force_include.difference(c.key for c in problem.candidates)
    if unknown:
        raise KeyError(f"强制包含的候选键不存在：{sorted(unknown)}")
    budget_nodes = node_budget or problem.search_node_budget
    candidates = problem.candidates

    # 强制包含项的即时约束冲突直接判不可行
    forced_state = ConstraintState.empty(problem)
    forced_info = InformationState(problem)
    for candidate in candidates:
        if candidate.key in force_include:
            reason = forced_state.block_reason(candidate)
            if reason is not None:
                return SolveResult(
                    method="exact",
                    status="infeasible",
                    selected=sorted(force_include),
                    cost=forced_state.cost,
                    information_gain=forced_info.information_gain,
                    upper_bound=None,
                    absolute_gap=None,
                    relative_gap=None,
                    nodes_visited=0,
                    node_budget=budget_nodes,
                    infeasibility_reason=f"强制包含方案不可行：{reason}",
                )
            forced_state.add(candidate)
            forced_info.add(candidate)

    # 阶段一：可行性骨架（仅约束 DFS），同时诊断无解
    skeleton, feasibility_nodes, reason = _feasibility_dfs(
        problem, budget_nodes, force_include
    )
    if skeleton is None:
        return SolveResult(
            method="exact",
            status="infeasible",
            selected=sorted(force_include),
            cost=forced_state.cost,
            information_gain=forced_info.information_gain,
            upper_bound=None,
            absolute_gap=None,
            relative_gap=None,
            nodes_visited=feasibility_nodes,
            node_budget=budget_nodes,
            infeasibility_reason=reason,
        )

    # 用骨架信息值作为初始 incumbent
    incumbent_state = skeleton
    incumbent_info = InformationState(problem)
    for key in skeleton.selected:
        incumbent_info.add(problem.candidate_by_key()[key])
    incumbent_gain = incumbent_info.information_gain

    counter = {"nodes": 0}
    stopped = {"flag": False}
    best_state: Optional[ConstraintState] = incumbent_state
    best_gain = incumbent_gain
    best_ub_seen = -math.inf

    def recurse(
        idx: int,
        constraints: ConstraintState,
        info: InformationState,
    ) -> None:
        nonlocal best_state, best_gain, best_ub_seen
        counter["nodes"] += 1
        if counter["nodes"] > budget_nodes:
            stopped["flag"] = True
            return

        remaining = candidates[idx:]
        remaining_groups = [c.group for c in remaining]
        if not constraints.minima_achievable(remaining_groups):
            return

        if idx == len(candidates):
            if constraints.minima_satisfied() and info.information_gain > best_gain + _TOL:
                best_state = constraints
                best_gain = info.information_gain
            return

        ub = combined_upper_bound(info, constraints, remaining)
        best_ub_seen = max(best_ub_seen, ub)
        if ub <= best_gain + _TOL:
            return  # 该子树不可能超过现任解

        candidate = candidates[idx]
        must_take = candidate.key in force_include

        # 包含分支（固定键次序；可行才进入）
        if constraints.block_reason(candidate) is None:
            taken_constraints = constraints.copy()
            taken_info = info.copy()
            taken_constraints.add(candidate)
            taken_info.add(candidate)
            recurse(idx + 1, taken_constraints, taken_info)
            if stopped["flag"]:
                return
        if not must_take:
            recurse(idx + 1, constraints.copy(), info.copy())

    recurse(0, ConstraintState.empty(problem), InformationState(problem))

    selected = list(best_state.selected)  # DFS 按固定键次序包含，天然有序
    final_info = InformationState(problem)
    steps: List[SelectionStep] = []
    cumulative_cost = 0.0
    for order, key in enumerate(selected, start=1):
        candidate = problem.candidate_by_key()[key]
        marginal = final_info.marginal_gain(candidate)
        cumulative_cost += candidate.cost
        final_info.add(candidate)
        steps.append(
            SelectionStep(
                order=order,
                key=key,
                marginal_gain=marginal,
                gain_per_cost=marginal / candidate.cost if candidate.cost > _TOL else math.inf,
                cost=candidate.cost,
                cumulative_gain=final_info.information_gain,
                cumulative_cost=cumulative_cost,
                phase="exact",
            )
        )

    root_ub = combined_upper_bound(
        InformationState(problem),
        ConstraintState.empty(problem),
        candidates,
    )
    status = "budget_stopped" if stopped["flag"] else "optimal"
    if stopped["flag"]:
        # 节点预算耗尽：保留松弛上界，报告与现任可行解的可证明 gap
        upper_bound = max(root_ub, best_ub_seen)
    else:
        # 全部子问题已枚举/剪枝：剪枝条件保证不存在更优可行解，
        # 现任解即全局最优，上界收敛到最优值，gap 为 0。
        upper_bound = best_gain
    gap = gap_values(best_gain, upper_bound)
    note = (
        "节点预算耗尽后停止：返回的是当前可行解，上界与 gap 可证明，"
        "不声称贪心近似比"
        if stopped["flag"]
        else "已枚举并剪枝全部子问题，方案在全部硬约束下最优"
    )
    return SolveResult(
        method="exact",
        status=status,
        selected=selected,
        cost=best_state.cost,
        information_gain=best_gain,
        upper_bound=upper_bound,
        absolute_gap=gap["absolute_gap"],
        relative_gap=gap["relative_gap"],
        nodes_visited=counter["nodes"] + feasibility_nodes,
        node_budget=budget_nodes,
        steps=steps,
        note=note,
    )


# --------------------------------------------------------------------------- #
# 可扩展增量搜索（可行性骨架 + 懒评估贪心）
# --------------------------------------------------------------------------- #
def _build_skeleton(
    problem: DesignProblem, node_budget: int
) -> Tuple[Optional[ConstraintState], int, Optional[str]]:
    return _feasibility_dfs(problem, node_budget)


def incremental_solve(
    problem: DesignProblem,
    force_include: Optional[set] = None,
    node_budget: Optional[int] = None,
) -> SolveResult:
    """增量搜索：先保证组下限可行，再按边际/成本比懒评估贪心加入。"""
    force_include = set(force_include or ())
    budget_nodes = node_budget or problem.search_node_budget
    by_key = problem.candidate_by_key()

    forced = ConstraintState.empty(problem)
    info = InformationState(problem)
    for candidate in problem.candidates:  # 固定键次序处理强制项
        if candidate.key in force_include:
            reason = forced.block_reason(candidate)
            if reason is not None:
                return SolveResult(
                    method="incremental",
                    status="infeasible",
                    selected=sorted(force_include),
                    cost=forced.cost,
                    information_gain=info.information_gain,
                    upper_bound=None,
                    absolute_gap=None,
                    relative_gap=None,
                    nodes_visited=0,
                    node_budget=budget_nodes,
                    infeasibility_reason=f"强制包含方案不可行：{reason}",
                )
            forced.add(candidate)
            info.add(candidate)

    skeleton, nodes, reason = _build_skeleton(problem, budget_nodes)
    if skeleton is None:
        return SolveResult(
            method="incremental",
            status="infeasible",
            selected=sorted(force_include),
            cost=forced.cost,
            information_gain=info.information_gain,
            upper_bound=None,
            absolute_gap=None,
            relative_gap=None,
            nodes_visited=nodes,
            node_budget=budget_nodes,
            infeasibility_reason=reason,
        )

    constraints = ConstraintState.empty(problem)
    info = InformationState(problem)
    steps: List[SelectionStep] = []
    order = 0

    # 阶段一：骨架（固定键次序），保证全部组下限，空方案仅在无下限需求时可行
    for key in skeleton.selected:
        candidate = by_key[key]
        marginal = info.marginal_gain(candidate)
        constraints.add(candidate)
        info.add(candidate)
        order += 1
        steps.append(
            SelectionStep(
                order=order,
                key=key,
                marginal_gain=marginal,
                gain_per_cost=(
                    marginal / candidate.cost if candidate.cost > _TOL else math.inf
                ),
                cost=candidate.cost,
                cumulative_gain=info.information_gain,
                cumulative_cost=constraints.cost,
                phase="skeleton",
            )
        )

    # 阶段二：标准懒评估贪心。堆条目 (-效率比, 键, 缓存时的步数)。
    # 子模性保证旧边际是当前边际的上界：堆顶弹出后若缓存过期则按当前状态
    # 重估并重新入堆，直到弹出的是当前步数缓存的条目——它即当前最优选择。
    # 不做全堆排空重建，均摊复杂度接近 O(n log n) 次边际评估。
    selected_set = set(constraints.selected)
    heap: List[Tuple[float, str, int]] = []

    def _push_candidate(candidate: Candidate) -> None:
        marginal = info.marginal_gain(candidate)
        ratio = marginal / candidate.cost if candidate.cost > _TOL else math.inf
        heapq.heappush(heap, (-ratio, candidate.key, len(steps)))

    for candidate in problem.candidates:
        if candidate.key in selected_set:
            continue
        if constraints.block_reason(candidate) is None:
            _push_candidate(candidate)

    while heap:
        neg_ratio, key, cached_at = heapq.heappop(heap)
        candidate = by_key[key]
        if key in selected_set or constraints.block_reason(candidate) is not None:
            continue  # 失效条目：已选或新触发硬约束
        if cached_at != len(steps):
            # 缓存过期：按当前状态重估后重新入堆（旧值仅作上界）
            _push_candidate(candidate)
            continue
        # 当前步数缓存的堆顶：即当前最高效率比，并列按键字典序
        marginal = info.marginal_gain(candidate)
        constraints.add(candidate)
        info.add(candidate)
        selected_set.add(key)
        order += 1
        steps.append(
            SelectionStep(
                order=order,
                key=key,
                marginal_gain=marginal,
                gain_per_cost=(
                    marginal / candidate.cost if candidate.cost > _TOL else math.inf
                ),
                cost=candidate.cost,
                cumulative_gain=info.information_gain,
                cumulative_cost=constraints.cost,
                phase="greedy",
            )
        )
        # 堆中其余条目保持不动：它们的缓存步数现在落后于 len(steps)，
        # 弹出时会按需重估，无需排空重建。

    upper_bound = combined_upper_bound(
        InformationState(problem),
        ConstraintState.empty(problem),
        problem.candidates,
    )
    gap = gap_values(info.information_gain, upper_bound)
    return SolveResult(
        method="incremental",
        status="budget_stopped",  # 启发式：即使 gap 为 0 也不声称固定近似比
        selected=list(constraints.selected),
        cost=constraints.cost,
        information_gain=info.information_gain,
        upper_bound=upper_bound,
        absolute_gap=gap["absolute_gap"],
        relative_gap=gap["relative_gap"],
        nodes_visited=nodes,
        node_budget=budget_nodes,
        steps=steps,
        note=(
            "增量搜索为启发式：预算/约束停止时返回当前可行解与可证明上界；"
            "一般约束下不存在固定近似比，gap 是与松弛上界的差距而非近似保证"
        ),
    )


# --------------------------------------------------------------------------- #
# 未选候选解释：约束不可加入 vs 全局机会损失
# --------------------------------------------------------------------------- #
@dataclass
class ExclusionExplanation:
    key: str
    classification: str  # constraint_blocked | reoptimized | selected
    block_reason: Optional[str]
    marginal_at_final: float
    forced_selected: Optional[List[str]]
    forced_gain: Optional[float]
    opportunity_loss: Optional[float]
    opportunity_loss_upper: Optional[float]
    detail: str


def explain_exclusion(
    problem: DesignProblem,
    result: SolveResult,
    method: str = "exact",
    node_budget: Optional[int] = None,
) -> List[ExclusionExplanation]:
    """对每个未选候选给出区分性解释。

    * ``constraint_blocked``：在最终方案状态下因硬约束不可加入，给出具体原因；
    * ``reoptimized``：可加入但未选，强制包含后重新优化，报告全局机会损失
      ``f(S*) - f(S*_with_j)``。精确法给精确差值；增量法给差值与上界区间。
    当下边际增益仅作描述性展示，明确不作为全局结论。
    """
    by_key = problem.candidate_by_key()
    final_constraints = ConstraintState.empty(problem)
    final_info = InformationState(problem)
    for key in result.selected:
        candidate = by_key[key]
        final_constraints.add(candidate)
        final_info.add(candidate)

    explanations: List[ExclusionExplanation] = []
    for candidate in problem.candidates:
        if candidate.key in result.selected:
            explanations.append(
                ExclusionExplanation(
                    key=candidate.key,
                    classification="selected",
                    block_reason=None,
                    marginal_at_final=0.0,
                    forced_selected=None,
                    forced_gain=None,
                    opportunity_loss=None,
                    opportunity_loss_upper=None,
                    detail="已选候选",
                )
            )
            continue

        marginal = final_info.marginal_gain(candidate)
        reason = final_constraints.block_reason(candidate)
        if reason is not None:
            explanations.append(
                ExclusionExplanation(
                    key=candidate.key,
                    classification="constraint_blocked",
                    block_reason=reason,
                    marginal_at_final=marginal,
                    forced_selected=None,
                    forced_gain=None,
                    opportunity_loss=None,
                    opportunity_loss_upper=None,
                    detail=(
                        f"{reason}。其当下边际增益 {marginal:.6f} nats 仅描述"
                        "信息吸引力，不能改变硬约束结论"
                    ),
                )
            )
            continue

        # 可加入但未选：强制包含后重新优化（全局机会损失，而非当下边际）
        forced = exact_solve(
            problem,
            force_include={candidate.key},
            node_budget=node_budget or problem.search_node_budget,
        ) if method == "exact" else incremental_solve(
            problem,
            force_include={candidate.key},
            node_budget=node_budget or problem.search_node_budget,
        )
        if forced.status == "infeasible":
            loss = result.information_gain
            explanations.append(
                ExclusionExplanation(
                    key=candidate.key,
                    classification="constraint_blocked",
                    block_reason=forced.infeasibility_reason,
                    marginal_at_final=marginal,
                    forced_selected=None,
                    forced_gain=None,
                    opportunity_loss=None,
                    opportunity_loss_upper=None,
                    detail=(
                        "最终方案状态下可加入，但强制包含并重新优化时不存在可行"
                        f"方案：{forced.infeasibility_reason}"
                    ),
                )
            )
            continue

        loss = result.information_gain - forced.information_gain
        if method == "exact" and forced.status == "optimal" and result.status == "optimal":
            loss_upper: Optional[float] = loss
            detail = (
                f"约束上可加入；强制包含后重新优化的最优增益 "
                f"{forced.information_gain:.6f}，全局机会损失 {loss:.6f} nats"
                "（精确差值，非当下边际）"
            )
        else:
            # 启发式或预算停止：只能给区间 [max(0, loss_hat), UB - f_forced]
            root_ub = result.upper_bound if result.upper_bound is not None else math.inf
            loss_upper = max(0.0, root_ub - forced.information_gain)
            detail = (
                f"约束上可加入；强制包含后重新优化得增益 {forced.information_gain:.6f}，"
                f"估计机会损失 {max(0.0, loss):.6f}，可证明损失上界 {loss_upper:.6f} nats"
                "（区间估计；当下边际不能替代全局结论）"
            )
        explanations.append(
            ExclusionExplanation(
                key=candidate.key,
                classification="reoptimized",
                block_reason=None,
                marginal_at_final=marginal,
                forced_selected=list(forced.selected),
                forced_gain=forced.information_gain,
                opportunity_loss=max(0.0, loss),
                opportunity_loss_upper=loss_upper,
                detail=detail,
            )
        )
    return explanations
