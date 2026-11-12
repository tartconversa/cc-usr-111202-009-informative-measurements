"""命令行入口与面向实验员的报告。

三个子命令：

* ``exact``       小规模精确分支定界；
* ``incremental`` 可扩展增量搜索（懒评估贪心 + 可行性骨架）；
* ``compare``     两种方法并排比较，展示启发式与可证明上界的差距。

默认输出文本报告到标准输出；``--json`` 输出机器可读结果，``--output``
可同时写入 ``.workspace-output/``（输入 fixture 永不覆盖）。本工具只做
先验下的设计计算，不执行任何设备操作，也不读取观测值。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence

from . import analysis
from .design_model import DesignProblem, load_problem
from .solvers import (
    ExclusionExplanation,
    SelectionStep,
    SolveResult,
    exact_solve,
    explain_exclusion,
    incremental_solve,
)

_STATUS_TEXT = {
    "optimal": "最优（全部子问题已枚举/剪枝）",
    "budget_stopped": "预算停止（当前可行解 + 可证明上界与 gap）",
    "infeasible": "无可行方案",
}


def _fmt(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and math.isinf(value):
        return "∞"
    return f"{value:.{digits}f}"


def _step_lines(steps: Sequence[SelectionStep]) -> List[str]:
    lines = [
        "  次序 阶段       候选键            本次信息收益(nats)  收益/成本   累计收益   累计成本",
    ]
    for step in steps:
        ratio = "∞" if math.isinf(step.gain_per_cost) else f"{step.gain_per_cost:.6f}"
        lines.append(
            f"  {step.order:>4d} {step.phase:<10s} "
            f"{step.key:<16s} {step.marginal_gain:>18.6f} "
            f"{ratio:>10s} {step.cumulative_gain:>10.6f} {step.cumulative_cost:>10.3g}"
        )
    return lines


def _direction_lines(problem: DesignProblem, state) -> List[str]:
    lines = ["", "参数方向方差变化（先验 → 后验，相对下降越大说明该方向被测量覆盖越好）："]
    for item in analysis.direction_variances(problem, state):
        lines.append(
            f"  {item.label:<28s} {item.prior_variance:>10.6f} → "
            f"{item.posterior_variance:>10.6f}   下降 {item.relative_reduction * 100:>6.2f}%"
        )
    lines.append("")
    lines.append("后验主方向（特征分解；最后一个是信息最弱、最值得补测的方向）：")
    for item in analysis.principal_directions(state):
        vector = "(" + ", ".join(f"{v:+.4f}" for v in item.direction) + ")"
        lines.append(
            f"  {item.label:<34s} 方向 {vector}  方差 {item.posterior_variance:.6f}"
            f"（下降 {item.relative_reduction * 100:.2f}%）"
        )
    return lines


def _sensitivity_lines(points: Sequence[analysis.SensitivityPoint]) -> List[str]:
    lines = [
        "",
        "噪声估计敏感性（噪声标准差乘以倍数；同一选中方案的增益与后验如何变化）：",
        "  倍数    增益(nats)   增益(log2)   后验对角方差                 最大主方差   重新优化后方案",
    ]
    for point in points:
        diagonal = "[" + ", ".join(f"{v:.4f}" for v in point.posterior_diagonal) + "]"
        if point.selection_stable is None:
            stability = "—"
        elif point.selection_stable:
            stability = "方案不变"
        else:
            stability = "方案变为 " + ",".join(point.reoptimized_keys or [])
        lines.append(
            f"  {point.noise_scale:>5.2f}  {point.information_gain_nats:>11.6f} "
            f"{point.information_gain_log2:>11.6f}  {diagonal:<26s} "
            f"{point.largest_posterior_variance:>10.6f}   {stability}"
        )
    return lines


def _exclusion_lines(explanations: Sequence[ExclusionExplanation]) -> List[str]:
    lines = [
        "",
        "未选候选解释（区分“约束不可加入”与“重新优化后的全局机会损失”）：",
    ]
    for item in explanations:
        if item.classification == "selected":
            continue
        if item.classification == "constraint_blocked":
            lines.append(f"  [{item.key}] 约束不可加入：{item.block_reason or item.detail}")
        else:
            lines.append(
                f"  [{item.key}] 可加入但全局机会损失 {_fmt(item.opportunity_loss)}"
                f"（上界 {_fmt(item.opportunity_loss_upper)}）nats；"
                f"强制包含后的方案 {item.forced_selected}"
            )
        lines.append(f"      当下边际增益（描述量，非全局结论）：{item.marginal_at_final:.6f} nats")
    return lines


def render_text_report(
    problem: DesignProblem,
    result: SolveResult,
    explanations: Sequence[ExclusionExplanation],
) -> str:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("线性测量设计报告（仅先验信息；无观测值，不执行设备操作）")
    lines.append("=" * 78)
    lines.append(f"候选数 {problem.n}，参数维度 {problem.dim}，总预算 {problem.budget:g}")
    lines.append(
        "组下限 "
        + (", ".join(f"{g}>={v}" for g, v in sorted(problem.group_min.items())) or "无")
        + "；组上限 "
        + (", ".join(f"{g}<={v}" for g, v in sorted(problem.group_max.items())) or "无")
        + f"；位置重复上限 {problem.position_upper_bound}"
    )
    lines.append("")
    lines.append(f"求解方法：{result.method}    状态：{_STATUS_TEXT[result.status]}")
    if result.note:
        lines.append(f"说明：{result.note}")
    if result.status == "infeasible":
        lines.append("")
        lines.append(f"不可行原因：{result.infeasibility_reason}")
        lines.append("空方案只在满足全部组下限时可行；当前下限无法满足，故无可行设计。")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"选中方案（固定键次序）：{result.selected}")
    lines.append(f"总成本 {result.cost:g} / 预算 {problem.budget:g}")
    lines.append(
        f"信息增益 {result.information_gain:.6f} nats "
        f"= {result.information_gain / math.log(2):.6f} log2 单位"
    )
    lines.append(
        f"可证明上界 {_fmt(result.upper_bound)} nats；"
        f"绝对 gap {_fmt(result.absolute_gap)}，相对 gap {_fmt(result.relative_gap)}"
    )
    if result.method == "incremental":
        lines.append(
            "注意：一般约束下贪心没有固定近似比；gap 是与预算松弛上界的差距，"
            "不是近似保证。"
        )
    lines.append("")
    lines.append("逐次选择的信息收益：")
    lines.extend(_step_lines(result.steps))

    state = analysis.build_state(problem, result.selected)
    lines.extend(_direction_lines(problem, state))
    lines.extend(_sensitivity_lines(analysis.noise_sensitivity(problem, result.selected)))
    lines.extend(_exclusion_lines(explanations))

    # 帮助实验员直接读出“多测一次”与“覆盖新方向”的差别
    lines.append("")
    lines.append("解读提示：同位置重复测量只压低该方向方差（对数收益递减）；")
    lines.append("覆盖新特征方向会同时缩小后验协方差的最大主方差，主方向列表的")
    lines.append("最后一项若仍接近先验值，说明该参数方向几乎未被任何测量覆盖。")
    return "\n".join(lines)


def result_to_dict(result: SolveResult) -> dict:
    data = asdict(result)
    return data


def explanations_to_dict(explanations: Sequence[ExclusionExplanation]) -> List[dict]:
    return [asdict(item) for item in explanations]


def _run(
    problem: DesignProblem,
    method: str,
    explain: bool,
) -> tuple[SolveResult, List[ExclusionExplanation]]:
    if method == "exact":
        result = exact_solve(problem)
    else:
        result = incremental_solve(problem)
    if not result.is_feasible:
        return result, []
    explanations = (
        explain_exclusion(problem, result, method=method) if explain else []
    )
    return result, explanations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="先验协方差与独立噪声下的受约束线性测量设计工具（不执行设备操作）"
    )
    parser.add_argument("candidates_path", help="候选测量 JSON（可内嵌设计字段）")
    parser.add_argument(
        "design_path", nargs="?", default=None, help="可选的独立设计配置 JSON"
    )
    parser.add_argument(
        "--method",
        choices=("exact", "incremental", "compare"),
        default="exact",
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    parser.add_argument(
        "--output",
        default=None,
        help="把报告同时写入该文件（建议位于 .workspace-output/）",
    )
    parser.add_argument(
        "--no-explain",
        action="store_true",
        help="跳过未选候选的重新优化解释（大规模实例可加速）",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    problem = load_problem(args.candidates_path, args.design_path)

    if args.method == "compare":
        exact_result, exact_expl = _run(problem, "exact", not args.no_explain)
        inc_result, inc_expl = _run(problem, "incremental", not args.no_explain)
        if args.json:
            payload = {
                "problem": _problem_summary(problem),
                "exact": result_to_dict(exact_result),
                "incremental": result_to_dict(inc_result),
                "exact_exclusions": explanations_to_dict(exact_expl),
                "incremental_exclusions": explanations_to_dict(inc_expl),
            }
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            text = (
                render_text_report(problem, exact_result, exact_expl)
                + "\n\n"
                + render_text_report(problem, inc_result, inc_expl)
                + "\n\n"
                + _comparison_lines(exact_result, inc_result)
            )
    else:
        result, explanations = _run(problem, args.method, not args.no_explain)
        if args.json:
            payload = {
                "problem": _problem_summary(problem),
                "result": result_to_dict(result),
                "exclusions": explanations_to_dict(explanations),
                "posterior": _posterior_payload(problem, result),
            }
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            if result.is_feasible:
                text = render_text_report(problem, result, explanations)
            else:
                text = render_text_report(problem, result, [])

    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    return 0


def _problem_summary(problem: DesignProblem) -> dict:
    return {
        "n_candidates": problem.n,
        "dim": problem.dim,
        "budget": problem.budget,
        "group_min": problem.group_min,
        "group_max": problem.group_max,
        "position_upper_bound": problem.position_upper_bound,
        "source_candidates": problem.source_candidates,
        "source_design": problem.source_design,
        "observation_free": True,
    }


def _posterior_payload(problem: DesignProblem, result: SolveResult) -> Optional[dict]:
    if not result.is_feasible:
        return None
    state = analysis.build_state(problem, result.selected)
    return {
        "selected": result.selected,
        "posterior_covariance": state.covariance,
        "precision_matrix": state.precision_matrix(),
        "directions": [asdict(d) for d in analysis.direction_variances(problem, state)],
        "principal_directions": [asdict(d) for d in analysis.principal_directions(state)],
        "noise_sensitivity": [asdict(p) for p in analysis.noise_sensitivity(problem, result.selected)],
    }


def _comparison_lines(exact_result: SolveResult, inc_result: SolveResult) -> str:
    lines = ["方法对比："]
    for name, result in (("精确分支定界", exact_result), ("增量搜索", inc_result)):
        if result.is_feasible:
            lines.append(
                f"  {name:<10s} 状态={result.status:<14s} "
                f"方案={result.selected} 增益={result.information_gain:.6f} "
                f"上界={_fmt(result.upper_bound)} gap={_fmt(result.absolute_gap)}"
            )
        else:
            lines.append(f"  {name:<10s} 不可行：{result.infeasibility_reason}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
