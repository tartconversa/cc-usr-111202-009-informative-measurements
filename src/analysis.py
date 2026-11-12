"""选择后的后验分析与噪声敏感性。

输入一个方案（候选键序列）与问题，输出：

* 后验协方差矩阵与精度矩阵（Cholesky 因子还原）；
* 各参数坐标方向、候选特征方向上的先验/后验方差与相对下降；
* 后验协方差的主方向（最弱信息方向，Jacobi 特征分解）；
* 噪声估计敏感性：把所有候选噪声方差乘以 ``s^2`` 后，同一方案的增益与
  后验协方差如何变化，并额外给出重新优化后方案是否改变的对比。

全部计算只使用先验与候选特征，不使用观测值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .design_model import Candidate, DesignProblem
from .information_state import InformationState
from .linear_algebra import jacobi_eigh, matvec


@dataclass
class DirectionVariance:
    label: str
    direction: List[float]
    prior_variance: float
    posterior_variance: float
    relative_reduction: float


@dataclass
class SensitivityPoint:
    noise_scale: float
    information_gain_nats: float
    information_gain_log2: float
    posterior_diagonal: List[float]
    largest_posterior_variance: float
    selection_stable: Optional[bool]
    reoptimized_keys: Optional[List[str]]


def build_state(problem: DesignProblem, selected_keys: Sequence[str]) -> InformationState:
    """按给定键序列重建信息状态（键序列决定更新次序，结果与次序无关）。"""
    by_key = problem.candidate_by_key()
    state = InformationState(problem)
    for key in selected_keys:
        state.add(by_key[key])
    return state


def _normalized(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 1e-15:
        raise ValueError("零向量不能作为方向")
    return [v / norm for v in vector]


def direction_variances(
    problem: DesignProblem,
    state: InformationState,
    extra_directions: Optional[Dict[str, Sequence[float]]] = None,
) -> List[DirectionVariance]:
    """参数坐标方向与候选特征方向上的方差及相对下降。"""
    dim = problem.dim
    out: List[DirectionVariance] = []
    prior = problem.prior_covariance
    posterior = state.covariance

    def variance(matrix, direction):
        mv = matvec(matrix, direction)
        return sum(d * v for d, v in zip(direction, mv))

    for i in range(dim):
        axis = [1.0 if k == i else 0.0 for k in range(dim)]
        before = variance(prior, axis)
        after = variance(posterior, axis)
        out.append(
            DirectionVariance(
                label=f"参数坐标 theta_{i + 1}",
                direction=axis,
                prior_variance=before,
                posterior_variance=after,
                relative_reduction=1.0 - after / before,
            )
        )

    seen = set()
    for candidate in problem.candidates:
        if not any(abs(v) > 1e-15 for v in candidate.row):
            continue
        direction = _normalized(candidate.row)
        signature = tuple(round(v, 12) for v in direction)
        if signature in seen:
            continue
        seen.add(signature)
        before = variance(prior, direction)
        after = variance(posterior, direction)
        out.append(
            DirectionVariance(
                label=f"候选特征方向 {candidate.key}",
                direction=direction,
                prior_variance=before,
                posterior_variance=after,
                relative_reduction=1.0 - after / before,
            )
        )

    if extra_directions:
        for label, vector in extra_directions.items():
            direction = _normalized(vector)
            before = variance(prior, direction)
            after = variance(posterior, direction)
            out.append(
                DirectionVariance(
                    label=label,
                    direction=direction,
                    prior_variance=before,
                    posterior_variance=after,
                    relative_reduction=1.0 - after / before,
                )
            )
    return out


def principal_directions(state: InformationState) -> List[DirectionVariance]:
    """后验协方差特征分解：按方差升序，标出信息最弱方向。"""
    values, vectors = jacobi_eigh(state.covariance)
    prior_values, _ = jacobi_eigh(state.prior_covariance)
    out: List[DirectionVariance] = []
    for j, eigenvalue in enumerate(values):
        direction = [vectors[i][j] for i in range(len(vectors))]
        # 主方向上的先验方差：用先验在该方向的二次型
        prior_mv = matvec(state.prior_covariance, direction)
        prior_variance = sum(d * v for d, v in zip(direction, prior_mv))
        out.append(
            DirectionVariance(
                label=f"后验主方向 {j + 1}（特征值 {eigenvalue:.6g}）",
                direction=direction,
                prior_variance=prior_variance,
                posterior_variance=eigenvalue,
                relative_reduction=1.0 - eigenvalue / prior_variance,
            )
        )
    return out


def _scaled_problem(problem: DesignProblem, scale: float) -> DesignProblem:
    """噪声标准差乘以 scale（方差乘 scale^2），其余契约不变的副本问题。"""
    from .design_model import Candidate

    scaled_candidates = [
        Candidate(
            key=c.key,
            row=c.row,
            noise_variance=c.noise_variance * scale * scale,
            cost=c.cost,
            group=c.group,
            position=c.position,
            mutex=c.mutex,
        )
        for c in problem.candidates
    ]
    clone = DesignProblem(
        candidates=scaled_candidates,
        prior_covariance=[row[:] for row in problem.prior_covariance],
        budget=problem.budget,
        group_min=dict(problem.group_min),
        group_max=dict(problem.group_max),
        position_upper_bound=problem.position_upper_bound,
        search_node_budget=problem.search_node_budget,
        noise_scale_sensitivity=problem.noise_scale_sensitivity,
        tie_break=problem.tie_break,
    )
    return clone


def noise_sensitivity(
    problem: DesignProblem,
    selected_keys: Sequence[str],
    reoptimize: bool = True,
) -> List[SensitivityPoint]:
    """对配置中的每个噪声倍数评估同一方案，并可对比重新优化结果。"""
    points: List[SensitivityPoint] = []
    for scale in problem.noise_scale_sensitivity:
        scaled = _scaled_problem(problem, scale)
        state = build_state(scaled, selected_keys)
        diagonal = [state.covariance[i][i] for i in range(problem.dim)]
        eigenvalues, _ = jacobi_eigh(state.covariance)

        selection_stable: Optional[bool] = None
        reoptimized_keys: Optional[List[str]] = None
        if reoptimize:
            from .solvers import exact_solve, incremental_solve

            solver = exact_solve if scaled.n <= 18 else incremental_solve
            re_result = solver(scaled)
            if re_result.is_feasible:
                reoptimized_keys = list(re_result.selected)
                selection_stable = reoptimized_keys == list(selected_keys)

        points.append(
            SensitivityPoint(
                noise_scale=scale,
                information_gain_nats=state.information_gain,
                information_gain_log2=state.information_gain / math.log(2.0),
                posterior_diagonal=diagonal,
                largest_posterior_variance=eigenvalues[-1],
                selection_stable=selection_stable,
                reoptimized_keys=reoptimized_keys,
            )
        )
    return points


def nats_to_log2(value_nats: float) -> float:
    return value_nats / math.log(2.0)
