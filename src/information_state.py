"""信息状态与约束状态：方案评估的稳定增量更新。

信息侧（:class:`InformationState`）同时维护

* 精度矩阵的 Cholesky 因子 ``L``（``P = L L^T``），测量加入走 O(d^2) 秩一更新；
* 后验协方差 ``Sigma``，走 Sherman-Morrison 秩一更新；

两者给出的边际信息增益在测试中交叉核对，避免任一路径的实现错误。
全部量只依赖先验与候选特征行/噪声，不含任何观测值。

约束侧（:class:`ConstraintState`）维护预算、仪器组数量、位置重复数与
互斥标签，并区分 *加入即违规* 与 *下限已无法满足* 两类不可行。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .design_model import Candidate, DesignProblem
from .linear_algebra import (
    chol_rank1_update,
    cholesky,
    invert_spd,
    logdet_from_chol,
    matvec,
)

_COST_TOL = 1e-12


class InformationState:
    """方案 S 的信息状态：P(S)=Sigma0^{-1}+sum u u^T，u=a/sigma。"""

    def __init__(self, problem: DesignProblem):
        self.problem = problem
        self.prior_covariance = [row[:] for row in problem.prior_covariance]
        self.prior_precision = invert_spd(problem.prior_covariance)
        # 当前精度矩阵的 Cholesky 因子
        self.chol = cholesky(self.prior_precision)
        # 当前后验协方差（精度矩阵之逆），初始即先验协方差
        self.covariance = [row[:] for row in self.prior_covariance]
        self.logdet_prior_precision = logdet_from_chol(self.chol)
        self.logdet_precision = self.logdet_prior_precision
        self.information_gain = 0.0

    def copy(self) -> "InformationState":
        clone = InformationState.__new__(InformationState)
        clone.problem = self.problem
        clone.prior_covariance = self.prior_covariance
        clone.prior_precision = self.prior_precision
        clone.chol = [row[:] for row in self.chol]
        clone.covariance = [row[:] for row in self.covariance]
        clone.logdet_prior_precision = self.logdet_prior_precision
        clone.logdet_precision = self.logdet_precision
        clone.information_gain = self.information_gain
        return clone

    def _scaled(self, candidate: Candidate) -> List[float]:
        """信息行 u = a / sigma，使 u u^T = a a^T / sigma^2。"""
        sigma = math.sqrt(candidate.noise_variance)
        return [v / sigma for v in candidate.row]

    def marginal_gain(self, candidate: Candidate) -> float:
        """加入该候选的信息增益（nats）：0.5 log(1 + u^T Sigma u)。

        这是 *当下边际* 量，用于选择过程与展示；未选候选的全局结论必须用
        重新优化后的机会损失，不能用本值替代（见 :mod:`src.solvers`）。
        """
        u = self._scaled(candidate)
        sigma_u = matvec(self.covariance, u)
        utility = sum(ui * sui for ui, sui in zip(u, sigma_u))
        return 0.5 * math.log1p(utility)

    def add(self, candidate: Candidate) -> float:
        """加入一个候选，返回本次选择的信息收益（nats）。"""
        u = self._scaled(candidate)
        before = self.logdet_precision

        # 1) 精度侧：Cholesky 秩一更新 P' = P + u u^T
        self.chol = chol_rank1_update(self.chol, u, 1.0)
        self.logdet_precision = logdet_from_chol(self.chol)

        # 2) 协方差侧：Sherman-Morrison
        #    Sigma' = Sigma - (Sigma u)(Sigma u)^T / (1 + u^T Sigma u)
        sigma_u = matvec(self.covariance, u)
        utility = sum(ui * sui for ui, sui in zip(u, sigma_u))
        denom = 1.0 + utility
        n = len(u)
        for i in range(n):
            for j in range(n):
                self.covariance[i][j] -= (
                    sigma_u[i] * sigma_u[j] / denom
                )

        gain = 0.5 * (self.logdet_precision - before)
        self.information_gain += gain
        return gain

    def precision_matrix(self) -> List[List[float]]:
        """由当前 Cholesky 因子还原精度矩阵 P = L L^T。"""
        n = len(self.chol)
        out = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                out[i][j] = sum(
                    self.chol[i][k] * self.chol[j][k] for k in range(j + 1)
                )
                out[j][i] = out[i][j]
        return out


@dataclass
class ConstraintState:
    """方案 S 的硬约束计数状态。"""

    problem: DesignProblem
    cost: float = 0.0
    group_counts: Dict[str, int] = None  # type: ignore[assignment]
    position_counts: Dict[str, int] = None  # type: ignore[assignment]
    mutex_used: set = None  # type: ignore[assignment]
    selected: List[str] = None  # type: ignore[assignment]

    @classmethod
    def empty(cls, problem: DesignProblem) -> "ConstraintState":
        return cls(
            problem=problem,
            cost=0.0,
            group_counts={},
            position_counts={},
            mutex_used=set(),
            selected=[],
        )

    def copy(self) -> "ConstraintState":
        return ConstraintState(
            problem=self.problem,
            cost=self.cost,
            group_counts=dict(self.group_counts),
            position_counts=dict(self.position_counts),
            mutex_used=set(self.mutex_used),
            selected=list(self.selected),
        )

    def block_reason(self, candidate: Candidate) -> Optional[str]:
        """返回阻止该候选 *立即加入* 的硬约束原因；可加入返回 None。"""
        budget = self.problem.budget
        if self.cost + candidate.cost > budget + _COST_TOL:
            return (
                f"预算不可加入：加入后成本 {self.cost + candidate.cost:g} "
                f"超过总预算 {budget:g}"
            )
        cap = self.problem.group_max.get(candidate.group, 10**9)
        if self.group_counts.get(candidate.group, 0) >= cap:
            return f"仪器组不可加入：组 {candidate.group} 已达上限 {cap}"
        if (
            self.position_counts.get(candidate.position, 0)
            >= self.problem.position_upper_bound
        ):
            return (
                f"重复位置不可加入：位置 {candidate.position} 已达重复上限 "
                f"{self.problem.position_upper_bound}"
            )
        clash = candidate.mutex and self.mutex_used.intersection(candidate.mutex)
        if clash:
            return f"互斥不可加入：标签 {sorted(clash)} 已被已选测量占用"
        return None

    def add(self, candidate: Candidate) -> None:
        self.cost += candidate.cost
        self.group_counts[candidate.group] = (
            self.group_counts.get(candidate.group, 0) + 1
        )
        self.position_counts[candidate.position] = (
            self.position_counts.get(candidate.position, 0) + 1
        )
        self.mutex_used.update(candidate.mutex)
        self.selected.append(candidate.key)

    def minima_deficit(self) -> Dict[str, int]:
        """每个仪器组距离 group_min 还差几个测量（下限为 0 的组不产生缺口）。"""
        return {
            group: max(0, lower - self.group_counts.get(group, 0))
            for group, lower in self.problem.group_min.items()
            if lower > 0 and self.group_counts.get(group, 0) < lower
        }

    def minima_satisfied(self) -> bool:
        return not self.minima_deficit()

    def minima_achievable(self, available_groups: Sequence[str]) -> bool:
        """给定仍可考虑的候选所属组集合，剩余缺口是否还有可能补齐。

        缺口按 *当前计数* 计算，而可用池只含尚未处理的后缀候选，因此已选
        成员不会被重复计入可用池——深节点不会被误判为不可行。
        """
        available_count: Dict[str, int] = {}
        for group in available_groups:
            available_count[group] = available_count.get(group, 0) + 1
        for group, deficit in self.minima_deficit().items():
            if available_count.get(group, 0) < deficit:
                return False
        return True

    def cheapest_minima_cost(self, candidates: Sequence[Candidate]) -> Optional[float]:
        """满足全部 group_min 还需的最小追加成本（只统计当前可加入候选）。

        互斥/位置/组上限冲突的候选已被 :meth:`block_reason` 排除，因此返回值
        是真实可行追加成本的下界；可加入候选数量不足以补齐下限时返回 None。
        """
        extra = 0.0
        for group, deficit in self.minima_deficit().items():
            costs = sorted(
                c.cost
                for c in candidates
                if c.group == group and self.block_reason(c) is None
            )
            if len(costs) < deficit:
                return None
            extra += sum(costs[:deficit])
        return extra
