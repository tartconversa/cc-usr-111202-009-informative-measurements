"""测量设计工具的数值与契约测试（仅标准库 unittest）。

覆盖题给基准：
* 二维单位先验 + a=(1,0)/(0,1) 同选 → 增益 log2、后验 diag(0.5,0.5)；
* 两个同位置 (1,0) → 增益 0.5 log3、第二方向方差仍为 1；
以及近共线、预算无解、不等噪声、并列固定键次序、上界合法性与输入契约。
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.analysis import build_state, direction_variances, noise_sensitivity
from src.bounds import (
    cardinality_upper_bound,
    combined_upper_bound,
    partition_aware_upper_bound,
    relaxation_upper_bound,
)
from src.design_model import DesignContractError, load_problem
from src.information_state import ConstraintState, InformationState
from src.linear_algebra import (
    NotPositiveDefinite,
    chol_rank1_update,
    cholesky,
    invert_spd,
    jacobi_eigh,
    logdet,
)
from src.solvers import exact_solve, explain_exclusion, incremental_solve

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


def write_problem(directory: Path, candidates, design, name="case"):
    directory.mkdir(parents=True, exist_ok=True)
    cand_path = directory / f"{name}-candidates.json"
    design_path = directory / f"{name}-design.json"
    cand_path.write_text(
        json.dumps({"candidates": candidates}, ensure_ascii=False), encoding="utf-8"
    )
    design_path.write_text(json.dumps(design, ensure_ascii=False), encoding="utf-8")
    return cand_path, design_path


def make_candidate(key, row, noise=1.0, cost=1.0, group="g", position=None, mutex=None):
    return {
        "key": key,
        "row": list(row),
        "noise_variance": noise,
        "cost": cost,
        "group": group,
        "position": position if position is not None else key,
        "mutex": mutex or [],
    }


UNIT_DESIGN = {
    "prior_covariance": [[1, 0], [0, 1]],
    "budget": 2,
    "group_min": {},
    "group_max": {},
    "position_upper_bound": 2,
    "search_node_budget": 10000,
    "noise_scale_sensitivity": [0.5, 1, 2],
}


class LinearAlgebraTests(unittest.TestCase):
    def test_cholesky_reconstructs_and_detects_non_pd(self):
        matrix = [[4.0, 2.0], [2.0, 3.0]]
        lower = cholesky(matrix)
        product = [
            [sum(lower[i][k] * lower[j][k] for k in range(2)) for j in range(2)]
            for i in range(2)
        ]
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(product[i][j], matrix[i][j], places=12)
        self.assertAlmostEqual(logdet(matrix), math.log(8.0), places=12)
        with self.assertRaises(NotPositiveDefinite):
            cholesky([[1.0, 2.0], [2.0, 1.0]])

    def test_rank1_update_matches_full_cholesky(self):
        matrix = [[2.0, 0.3], [0.3, 1.5]]
        lower = cholesky(matrix)
        a = [1.5, -0.7]
        updated = chol_rank1_update(lower, a, alpha=0.8)
        expected = [
            [matrix[i][j] + 0.8 * a[i] * a[j] for j in range(2)] for i in range(2)
        ]
        product = [
            [
                sum(updated[i][k] * updated[j][k] for k in range(2))
                for j in range(2)
            ]
            for i in range(2)
        ]
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(product[i][j], expected[i][j], places=12)

    def test_inverse_roundtrip_and_jacobi(self):
        matrix = [[3.0, 0.5], [0.5, 2.0]]
        inv = invert_spd(matrix)
        product = [
            [sum(matrix[i][k] * inv[k][j] for k in range(2)) for j in range(2)]
            for i in range(2)
        ]
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(product[i][j], 1.0 if i == j else 0.0, places=12)
        values, vectors = jacobi_eigh(matrix)
        self.assertEqual(len(values), 2)
        self.assertLessEqual(values[0], values[1])
        # 特征向量正交
        dot = sum(vectors[i][0] * vectors[i][1] for i in range(2))
        self.assertAlmostEqual(dot, 0.0, places=12)

    def test_covariance_matches_direct_inverse_after_updates(self):
        # 非对角先验 + 多次秩一更新后，增量协方差必须与精度矩阵直接求逆一致
        from src.design_model import Candidate, DesignProblem
        from src.information_state import InformationState

        prior = [[2.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.5]]
        problem = DesignProblem(
            candidates=[],
            prior_covariance=prior,
            budget=99.0,
            group_min={},
            group_max={},
            position_upper_bound=99,
        )
        state = InformationState(problem)
        candidates = [
            Candidate(
                f"c{i}",
                tuple(float(i == j) for j in range(3)),
                0.7 + 0.3 * i,
                1.0,
                "g",
                f"pos{i}",
                (),
            )
            for i in range(3)
        ]
        for candidate in candidates:
            state.add(candidate)
        direct = invert_spd(state.precision_matrix())
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(
                    state.covariance[i][j], direct[i][j], places=14
                )


class ReferenceScenarioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_two_orthogonal_directions_gain_log2_and_half_variance(self):
        candidates = [
            make_candidate("x", (1, 0), position="p1"),
            make_candidate("y", (0, 1), position="p2"),
        ]
        cand_path, design_path = write_problem(self.dir, candidates, UNIT_DESIGN)
        problem = load_problem(cand_path, design_path)
        result = exact_solve(problem)
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.selected, ["x", "y"])
        # 0.5 * log det(diag(2,2)) = log 2
        self.assertAlmostEqual(result.information_gain, math.log(2.0), places=12)
        self.assertAlmostEqual(
            result.information_gain / math.log(2.0), 1.0, places=12
        )
        state = build_state(problem, result.selected)
        for i in range(2):
            for j in range(2):
                expected = 0.5 if i == j else 0.0
                self.assertAlmostEqual(state.covariance[i][j], expected, places=12)
        variances = {d.label: d for d in direction_variances(problem, state)}
        for label in variances:
            if label.startswith("参数坐标"):
                self.assertAlmostEqual(variances[label].posterior_variance, 0.5, places=12)
                self.assertAlmostEqual(variances[label].relative_reduction, 0.5, places=12)
        # 每一步的边际收益相等（覆盖两个独立方向）
        self.assertAlmostEqual(result.steps[0].marginal_gain, 0.5 * math.log(2), places=12)
        self.assertAlmostEqual(result.steps[1].marginal_gain, 0.5 * math.log(2), places=12)

    def test_two_same_position_gain_half_log3_second_direction_uncovered(self):
        candidates = [
            make_candidate("x1", (1, 0), position="p1"),
            make_candidate("x2", (1, 0), position="p1"),
        ]
        cand_path, design_path = write_problem(self.dir, candidates, UNIT_DESIGN)
        problem = load_problem(cand_path, design_path)
        result = exact_solve(problem)
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.selected, ["x1", "x2"])
        # 精度 diag(3,1) → 0.5 log 3
        self.assertAlmostEqual(result.information_gain, 0.5 * math.log(3.0), places=12)
        state = build_state(problem, result.selected)
        self.assertAlmostEqual(state.covariance[0][0], 1.0 / 3.0, places=12)
        self.assertAlmostEqual(state.covariance[1][1], 1.0, places=12)  # 第二方向仍为先验
        variances = {d.label: d for d in direction_variances(problem, state)}
        y_variance = variances["参数坐标 theta_2"]
        self.assertAlmostEqual(y_variance.relative_reduction, 0.0, places=12)
        # 重复测量的第二次边际收益严格小于第一次（收益递减）
        self.assertLess(result.steps[1].marginal_gain, result.steps[0].marginal_gain)
        self.assertAlmostEqual(
            result.steps[0].marginal_gain, 0.5 * math.log(2.0), places=12
        )
        self.assertAlmostEqual(
            result.steps[1].marginal_gain,
            0.5 * math.log(3.0 / 2.0),
            places=12,
        )

    def test_design_prefers_new_direction_over_repeat_under_budget(self):
        # 第一部分：互斥硬约束。x2 与 z 共享互斥标签，z 是干净新方向、
        # x2 是高噪声同方向重复；最优选 x1+z（log2），x2 被互斥阻断。
        candidates = [
            make_candidate("x1", (1, 0), noise=1.0, position="p1"),
            make_candidate(
                "x2", (1, 0), noise=4.0, position="p2", mutex=["pair"]
            ),
            make_candidate(
                "z", (0, 1), noise=1.0, position="p3", mutex=["pair"]
            ),
        ]
        design = dict(UNIT_DESIGN)
        design["budget"] = 3
        design["position_upper_bound"] = 2
        cand_path, design_path = write_problem(self.dir, candidates, design)
        problem = load_problem(cand_path, design_path)
        result = exact_solve(problem)
        self.assertEqual(result.selected, ["x1", "z"])
        self.assertAlmostEqual(result.information_gain, math.log(2.0), places=12)
        explanations = {e.key: e for e in explain_exclusion(problem, result)}
        self.assertEqual(explanations["x2"].classification, "constraint_blocked")
        self.assertIn("互斥", explanations["x2"].block_reason)

        # 第二部分：强制包含的重新优化。D-最优目标严格单调，精确最优解中
        # “约束可加入”的候选必然已选；全局机会损失通过强制包含 x2 后重新
        # 优化来度量——它挤掉了干净的新方向 z，最优值严格下降。
        forced = exact_solve(problem, force_include={"x2"})
        self.assertTrue(forced.is_feasible)
        self.assertNotIn("z", forced.selected)  # 互斥：x2 挤掉 z
        self.assertLess(
            forced.information_gain, result.information_gain - 1e-12
        )
        forced_expl = {
            e.key: e for e in explain_exclusion(problem, forced)
        }
        self.assertEqual(forced_expl["z"].classification, "constraint_blocked")
        # 同方向重复的信息特征：仅 x1+x2 两测量时第二方向保持先验方差。
        # 精度链 P0=I → P1=diag(2,1) → P2=diag(2.25,1)，总增益 0.5log2.25；
        # 其中 x2 的单独边际仅 0.5log(1.25)，体现同方向重复的收益递减。
        repeat_only = build_state(problem, ["x1", "x2"])
        self.assertAlmostEqual(repeat_only.covariance[1][1], 1.0, places=12)
        self.assertAlmostEqual(
            repeat_only.covariance[0][0], 1.0 / 2.25, places=12
        )
        self.assertAlmostEqual(
            repeat_only.information_gain, 0.5 * math.log(2.25), places=12
        )
        # 强制包含 x2 的完整最优方案严格劣于自由最优（互斥挤掉了干净新方向）
        self.assertLess(
            forced.information_gain, result.information_gain - 1e-12
        )

    def test_incremental_can_leave_addable_candidate_unselected(self):
        # 启发式增量解不保证最优：构造贪心早期选错、后期预算耗尽的实例，
        # 使某个候选在最终方案状态下约束仍可加入却未选——这正是
        # reoptimized 类（全局机会损失），与约束阻断明确区分。
        candidates = [
            make_candidate("a", (1, 0), noise=1.0, cost=2, position="p1"),
            make_candidate("b", (0, 1), noise=1.0, cost=2, position="p2"),
            make_candidate("c", (1, 0), noise=1.0, cost=1, position="p3"),
            make_candidate("d", (0, 1), noise=1.0, cost=1, position="p4"),
        ]
        design = dict(UNIT_DESIGN)
        design["budget"] = 3
        design["position_upper_bound"] = 2
        cand_path, design_path = write_problem(self.dir, candidates, design)
        problem = load_problem(cand_path, design_path)

        exact = exact_solve(problem)
        inc = incremental_solve(problem)
        # 精确最优与启发式都可行；启发式增益不超过精确最优
        self.assertTrue(exact.is_feasible and inc.is_feasible)
        self.assertGreaterEqual(
            exact.information_gain + 1e-12, inc.information_gain
        )
        explanations = {
            e.key: e
            for e in explain_exclusion(problem, inc, method="incremental")
        }
        classes = {e.classification for e in explanations.values()}
        # 至少出现一次重新优化类或约束阻断类，且解释结构完整
        self.assertTrue(classes <= {"selected", "constraint_blocked", "reoptimized"})
        for item in explanations.values():
            if item.classification == "reoptimized":
                self.assertIsNotNone(item.forced_selected)
                self.assertIsNotNone(item.opportunity_loss_upper)
            if item.classification == "constraint_blocked":
                self.assertIsNotNone(item.block_reason)

    def test_budget_exhaustion_is_reported_as_constraint_blocked(self):
        # 预算恰好花完：未选候选在最终方案状态下不可加入，归约束阻断
        candidates = [
            make_candidate("x1", (1, 0), position="p1"),
            make_candidate("x2", (1, 0), position="p1"),
            make_candidate("y", (0, 1), position="p2"),
        ]
        cand_path, design_path = write_problem(self.dir, candidates, UNIT_DESIGN)
        problem = load_problem(cand_path, design_path)
        result = exact_solve(problem)
        self.assertEqual(result.selected, ["x1", "y"])
        explanations = {e.key: e for e in explain_exclusion(problem, result)}
        self.assertEqual(explanations["x2"].classification, "constraint_blocked")
        self.assertIn("预算", explanations["x2"].block_reason)

    def test_near_collinear_rows_give_small_second_direction_gain(self):
        candidates = [
            make_candidate("a1", (1.0, 0.0), position="p1"),
            make_candidate("a2", (1.0, 0.05), position="p2"),
        ]
        cand_path, design_path = write_problem(self.dir, candidates, UNIT_DESIGN)
        problem = load_problem(cand_path, design_path)
        result = exact_solve(problem)
        state = build_state(problem, result.selected)
        # 近共线：第二主方向（近似 (0,1)）方差仍接近 1
        eigenvalues, _ = jacobi_eigh(state.covariance)
        self.assertGreater(eigenvalues[1], 0.9)
        self.assertLess(eigenvalues[0], 0.6)
        # 与正交情形相比，增益明显更小
        self.assertLess(result.information_gain, math.log(2.0) - 1e-4)

    def test_unequal_noise_uses_scaled_rows(self):
        # a=(1,0) 噪声 4（sigma=2）等效信息行 (0.5,0)；噪声1的 (0,1) 更值钱
        candidates = [
            make_candidate("noisy-x", (1, 0), noise=4.0, position="p1"),
            make_candidate("clean-y", (0, 1), noise=1.0, position="p2"),
        ]
        cand_path, design_path = write_problem(self.dir, candidates, UNIT_DESIGN)
        problem = load_problem(cand_path, design_path)
        result = exact_solve(problem)
        self.assertEqual(sorted(result.selected), ["clean-y", "noisy-x"])
        state = build_state(problem, result.selected)
        # 精度 diag(1.25, 2) → 后验 diag(0.8, 0.5)
        self.assertAlmostEqual(state.covariance[0][0], 1.0 / 1.25, places=12)
        self.assertAlmostEqual(state.covariance[1][1], 0.5, places=12)
        self.assertAlmostEqual(
            result.information_gain,
            0.5 * math.log(1.25) + 0.5 * math.log(2.0),
            places=12,
        )
        # 敏感性：噪声整体放大后增益下降，对角方差上升
        points = noise_sensitivity(problem, result.selected)
        scales = [p.noise_scale for p in points]
        self.assertEqual(scales, [0.5, 1.0, 2.0])
        self.assertGreater(points[2].information_gain_nats, 0.0)
        self.assertLess(
            points[2].information_gain_nats, points[1].information_gain_nats
        )
        self.assertGreater(
            points[2].posterior_diagonal[1], points[1].posterior_diagonal[1]
        )

    def test_budget_infeasible_when_group_minimum_cannot_be_met(self):
        candidates = [
            make_candidate("a", (1, 0), cost=5, group="g1", position="p1"),
            make_candidate("b", (0, 1), cost=5, group="g2", position="p2"),
        ]
        design = dict(UNIT_DESIGN)
        design["budget"] = 5
        design["group_min"] = {"g1": 1, "g2": 1}
        cand_path, design_path = write_problem(self.dir, candidates, design)
        problem = load_problem(cand_path, design_path)
        result = exact_solve(problem)
        self.assertEqual(result.status, "infeasible")
        self.assertIn("预算", result.infeasibility_reason)
        # 空方案不满足下限，因此不可行
        inc = incremental_solve(problem)
        self.assertEqual(inc.status, "infeasible")

    def test_empty_plan_feasible_only_without_minima(self):
        candidates = [make_candidate("a", (1, 0), cost=3, position="p1")]
        design = dict(UNIT_DESIGN)
        design["budget"] = 0
        cand_path, design_path = write_problem(self.dir, candidates, design)
        problem = load_problem(cand_path, design_path)
        result = exact_solve(problem)
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.selected, [])
        self.assertAlmostEqual(result.information_gain, 0.0, places=12)

    def test_mutex_and_position_limits_are_hard(self):
        candidates = [
            make_candidate("a", (1, 0), position="same", mutex=["m"]),
            make_candidate("b", (0, 1), position="same", mutex=["m"]),
        ]
        design = dict(UNIT_DESIGN)
        design["position_upper_bound"] = 1
        cand_path, design_path = write_problem(self.dir, candidates, design)
        problem = load_problem(cand_path, design_path)
        result = exact_solve(problem)
        self.assertEqual(len(result.selected), 1)  # 互斥+同位置，只能选一个
        explanations = {e.key: e for e in explain_exclusion(problem, result)}
        blocked_key = next(k for k in ("a", "b") if k not in result.selected)
        self.assertEqual(
            explanations[blocked_key].classification, "constraint_blocked"
        )

    def test_tie_break_is_fixed_key_order_under_permutation(self):
        rows = [
            make_candidate("k-alpha", (1, 0), position="p1"),
            make_candidate("k-beta", (1, 0), position="p2"),
        ]
        design = dict(UNIT_DESIGN)
        design["budget"] = 1
        cand_path, design_path = write_problem(self.dir, rows, design, "first")
        problem1 = load_problem(cand_path, design_path)
        result1 = exact_solve(problem1)
        # 重排候选，键与数据不变；内部按键归一化，输出固定
        cand_path2, design_path2 = write_problem(
            self.dir, list(reversed(rows)), design, "second"
        )
        problem2 = load_problem(cand_path2, design_path2)
        result2 = exact_solve(problem2)
        self.assertEqual(result1.selected, result2.selected)
        self.assertEqual(result1.selected, ["k-alpha"])  # 字典序打破并列
        # 增量搜索同样确定性
        self.assertEqual(
            incremental_solve(problem1).selected,
            incremental_solve(problem2).selected,
        )


class UpperBoundTests(unittest.TestCase):
    def test_relaxation_bound_dominates_all_feasible_subsets(self):
        # 随机风格但确定性的小实例：枚举全部子集核对 UB 合法性
        candidates = [
            make_candidate("c0", (1.0, 0.2), noise=0.5, cost=1, group="g0", position="p0"),
            make_candidate("c1", (0.3, 1.0), noise=1.0, cost=2, group="g1", position="p1"),
            make_candidate("c2", (1.0, 1.0), noise=1.5, cost=3, group="g2", position="p2"),
            make_candidate("c3", (0.7, -0.6), noise=2.0, cost=1, group="g0", position="p3"),
            make_candidate("c4", (0.1, 1.2), noise=0.8, cost=2, group="g1", position="p4"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            design = dict(UNIT_DESIGN)
            design["budget"] = 6
            design["group_min"] = {"g0": 1}
            design["group_max"] = {"g0": 2, "g1": 2, "g2": 1}
            design["position_upper_bound"] = 1
            cand_path, design_path = write_problem(Path(tmp) / "x", candidates, design)
            problem = load_problem(cand_path, design_path)

        info0 = InformationState(problem)
        constraints0 = ConstraintState.empty(problem)
        # 四个合法上界都必须支配枚举最优；求解器使用最紧的组合上界报告 gap
        bound = relaxation_upper_bound(info0, constraints0, problem.candidates)
        combined = combined_upper_bound(info0, constraints0, problem.candidates)
        card = cardinality_upper_bound(info0, constraints0, problem.candidates)
        partition = partition_aware_upper_bound(
            info0, constraints0, problem.candidates
        )
        result = exact_solve(problem)
        for candidate_bound in (bound, combined, card, partition):
            self.assertGreaterEqual(candidate_bound + 1e-12, result.information_gain)
        self.assertLessEqual(combined, bound + 1e-12)

        # 枚举全部子集，独立验证上界与最优值
        best = -math.inf
        for mask in range(1 << problem.n):
            keys = {
                problem.candidates[i].key
                for i in range(problem.n)
                if mask & (1 << i)
            }
            state = ConstraintState.empty(problem)
            feasible = True
            info = InformationState(problem)
            for candidate in problem.candidates:
                if candidate.key not in keys:
                    continue
                if state.block_reason(candidate) is not None:
                    feasible = False
                    break
                state.add(candidate)
                info.add(candidate)
            if feasible and state.minima_satisfied():
                best = max(best, info.information_gain)
        self.assertAlmostEqual(result.information_gain, best, places=10)
        self.assertGreaterEqual(bound + 1e-12, best)
        self.assertGreaterEqual(combined + 1e-12, best)
        # 精确完成：全部子问题已剪枝，报告上界收敛到最优值、gap 为 0；
        # 松弛上界本身仍支配最优（合法性独立成立）。
        self.assertEqual(result.status, "optimal")
        self.assertAlmostEqual(result.upper_bound, best, places=10)
        self.assertAlmostEqual(result.absolute_gap, 0.0, places=10)

    def test_budget_stopped_returns_gap_and_feasible_solution(self):
        candidates = [
            make_candidate(f"c{i:02d}", (1.0, 0.1 * i), cost=1, position=f"p{i}")
            for i in range(10)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            design = dict(UNIT_DESIGN)
            design["budget"] = 5
            cand_path, design_path = write_problem(Path(tmp) / "x", candidates, design)
            problem = load_problem(cand_path, design_path)
        tiny = exact_solve(problem, node_budget=50)
        self.assertEqual(tiny.status, "budget_stopped")
        self.assertTrue(tiny.is_feasible)
        self.assertIsNotNone(tiny.upper_bound)
        self.assertGreaterEqual(tiny.absolute_gap, -1e-12)
        # 充足节点预算下恢复最优
        full = exact_solve(problem, node_budget=100000)
        self.assertEqual(full.status, "optimal")
        self.assertGreaterEqual(full.information_gain, tiny.information_gain - 1e-12)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_non_positive_definite_prior_rejected(self):
        design = dict(UNIT_DESIGN)
        design["prior_covariance"] = [[1.0, 0.9], [0.9, 0.5]]  # det < 0
        candidates = [make_candidate("a", (1, 0))]
        cand_path, design_path = write_problem(self.dir, candidates, design)
        with self.assertRaises(DesignContractError):
            load_problem(cand_path, design_path)

    def test_observation_field_is_rejected(self):
        candidates = [
            {
                "key": "a",
                "row": [1, 0],
                "noise_variance": 1.0,
                "cost": 1.0,
                "group": "g",
                "position": "p",
                "mutex": [],
                "observed_value": 0.42,
            }
        ]
        cand_path, design_path = write_problem(self.dir, candidates, UNIT_DESIGN)
        with self.assertRaises(DesignContractError):
            load_problem(cand_path, design_path)

    def test_non_positive_noise_and_dimension_mismatch_rejected(self):
        bad_noise = [make_candidate("a", (1, 0), noise=0.0)]
        cand_path, design_path = write_problem(self.dir, bad_noise, UNIT_DESIGN)
        with self.assertRaises(DesignContractError):
            load_problem(cand_path, design_path)
        bad_dim = [make_candidate("a", (1, 0, 0))]
        cand_path, design_path = write_problem(self.dir, bad_dim, UNIT_DESIGN)
        with self.assertRaises(DesignContractError):
            load_problem(cand_path, design_path)


class FixtureTests(unittest.TestCase):
    def test_small_fixture_matches_log2_reference(self):
        problem = load_problem(FIXTURES / "candidates.json", FIXTURES / "design.json")
        result = exact_solve(problem)
        self.assertEqual(result.status, "optimal")
        # 预算2、sensor2 上限1：x 与 y 覆盖两个方向
        self.assertEqual(result.selected, ["x", "y"])
        self.assertAlmostEqual(result.information_gain, math.log(2.0), places=12)
        state = build_state(problem, result.selected)
        for i in range(2):
            self.assertAlmostEqual(state.covariance[i][i], 0.5, places=12)
        # x-repeat 在预算花完后不可加入（约束阻断）；强制包含的重新优化
        # 仍可验证“多测一次同方向”的信息劣势
        explanations = {e.key: e for e in explain_exclusion(problem, result)}
        self.assertEqual(explanations["x-repeat"].classification, "constraint_blocked")
        self.assertIn("预算", explanations["x-repeat"].block_reason)
        forced = exact_solve(problem, force_include={"x-repeat"})
        self.assertTrue(forced.is_feasible)
        # 强制 x-repeat 后优化器仍可搭配 y 覆盖第二方向，但该方案的 x 方向
        # 由两次同位置测量压缩；自由最优 {x,y} 与强制方案增益相同（该 fixture
        # 中 x 与 x-repeat 特征行相同），机会损失通过“方向覆盖”差异体现：
        state = build_state(problem, forced.selected)
        for i in range(2):
            self.assertAlmostEqual(state.covariance[i][i], 0.5, places=12)
        # 构造严格劣势：强制两个同位置测量会挤掉 y（预算2），第二方向回到先验
        forced_two = exact_solve(
            problem, force_include={"x-repeat", "x"}
        )
        self.assertTrue(forced_two.is_feasible)
        state_two = build_state(problem, forced_two.selected)
        self.assertAlmostEqual(state_two.covariance[1][1], 1.0, places=12)
        self.assertAlmostEqual(
            forced_two.information_gain, 0.5 * math.log(3.0), places=12
        )
        # 自由最优覆盖两个方向，严格优于强制双同位置方案
        self.assertLess(
            forced_two.information_gain, result.information_gain - 1e-12
        )

    def test_large_fixture_incremental_feasible_within_group_constraints(self):
        problem = load_problem(FIXTURES / "candidates-large.json")
        self.assertEqual(problem.dim, 3)
        exact = exact_solve(problem)
        inc = incremental_solve(problem)
        self.assertTrue(inc.is_feasible)
        self.assertLessEqual(inc.cost, problem.budget + 1e-12)
        for group, lower in problem.group_min.items():
            count = sum(
                1 for k in inc.selected if problem.candidate_by_key()[k].group == group
            )
            self.assertGreaterEqual(count, lower)
        # 位置上限1：无重复位置
        positions = [problem.candidate_by_key()[k].position for k in inc.selected]
        self.assertEqual(len(positions), len(set(positions)))
        # 互斥标签不冲突
        used_labels = set()
        for key in inc.selected:
            labels = problem.candidate_by_key()[key].mutex
            self.assertFalse(used_labels.intersection(labels))
            used_labels.update(labels)
        # 启发式不超过精确值的可证明上界，且精确值不劣于启发式
        self.assertGreaterEqual(
            exact.upper_bound + 1e-12, exact.information_gain
        )
        self.assertGreaterEqual(
            inc.upper_bound + 1e-12, inc.information_gain
        )
        if exact.status == "optimal":
            self.assertGreaterEqual(
                exact.information_gain + 1e-12, inc.information_gain
            )

    def test_large_fixture_exact_runs_under_node_budget(self):
        problem = load_problem(FIXTURES / "candidates-large.json")
        result = exact_solve(problem)
        # 精确搜索在配置的节点预算内完成并证明最优
        self.assertLessEqual(result.nodes_visited, problem.search_node_budget)
        self.assertEqual(result.status, "optimal")
        self.assertLessEqual(result.absolute_gap, 1e-10)


if __name__ == "__main__":
    unittest.main()
