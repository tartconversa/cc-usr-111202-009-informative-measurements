"""对称正定线性代数（仅标准库实现）。

所有例程针对对称正定矩阵；精度矩阵/协方差矩阵均满足该性质：

* :func:`cholesky`        —— 稳定 Cholesky 分解，非正定抛 ``NotPositiveDefinite``；
* :func:`chol_rank1_update` —— 秩一更新 ``P + alpha a a^T`` 的 Cholesky 因子，
  对应一次测量加入后的稳定增量更新；
* :func:`logdet_from_chol` —— 由 Cholesky 因子计算 ``log det``；
* :func:`invert_spd`       —— 对称正定求逆（后验协方差）；
* :func:`jacobi_eigh`      —— 对称矩阵 Jacobi 特征分解（主方向）。

不依赖任何第三方库，浮点判定使用相对容差，避免把近奇异矩阵误判为正定。
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


class NotPositiveDefinite(ValueError):
    """矩阵在给定容差下不是对称正定矩阵。"""


def _zero(n: int) -> List[List[float]]:
    return [[0.0 for _ in range(n)] for _ in range(n)]


def identity(n: int) -> List[List[float]]:
    out = _zero(n)
    for i in range(n):
        out[i][i] = 1.0
    return out


def copy_matrix(a: Matrix) -> List[List[float]]:
    return [list(map(float, row)) for row in a]


def symmetric(a: Matrix, tol: float = 1e-12) -> bool:
    n = len(a)
    scale = max(
        (abs(a[i][j]) for i in range(n) for j in range(n)),
        default=1.0,
    )
    eps = tol * max(1.0, scale)
    return all(abs(a[i][j] - a[j][i]) <= eps for i in range(n) for j in range(n))


def check_symmetric(a: Matrix, name: str = "matrix") -> None:
    if not a or any(len(row) != len(a) for row in a):
        raise ValueError(f"{name} 必须是方阵")
    if not symmetric(a):
        raise ValueError(f"{name} 必须对称")


def cholesky(a: Matrix, tol: float = 1e-12) -> List[List[float]]:
    """返回下三角 L 使 ``L L^T = a``；非正定抛 :class:`NotPositiveDefinite`。"""
    check_symmetric(a, "Cholesky 输入")
    n = len(a)
    scale = max(
        (abs(a[i][j]) for i in range(n) for j in range(n)),
        default=1.0,
    )
    eps = tol * max(1.0, scale)
    lower = _zero(n)
    for i in range(n):
        for j in range(i + 1):
            value = a[i][j] - sum(
                lower[i][k] * lower[j][k] for k in range(j)
            )
            if i == j:
                if value <= eps:
                    raise NotPositiveDefinite(
                        f"Cholesky 失败：主元 {value:.3e} 非正，矩阵非正定"
                    )
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value / lower[j][j]
    return lower


def chol_rank1_update(
    lower: Matrix,
    a: Vector,
    alpha: float = 1.0,
) -> List[List[float]]:
    """秩一更新：返回 ``L'`` 使 ``L' L'^T = L L^T + alpha a a^T``。

    纯加法更新（``alpha > 0``，测量只会增加信息），算法为逐分量消去，
    等价于 Cholesky 但只需 O(n^2)，数值上不重新做完整分解。
    """
    if alpha <= 0.0:
        raise ValueError("秩一更新要求 alpha > 0")
    n = len(lower)
    new_l = copy_matrix(lower)
    x = [math.sqrt(alpha) * float(v) for v in a]
    for k in range(n):
        r = math.hypot(new_l[k][k], x[k])
        c = r / new_l[k][k]
        s = x[k] / new_l[k][k]
        new_l[k][k] = r
        if k + 1 < n:
            for i in range(k + 1, n):
                new_l[i][k] = (new_l[i][k] + s * x[i]) / c
                x[i] = c * x[i] - s * new_l[i][k]
    return new_l


def logdet_from_chol(lower: Matrix) -> float:
    """``log det A = 2 sum log L_ii``。"""
    return 2.0 * sum(math.log(lower[i][i]) for i in range(len(lower)))


def solve_triangular(lower: Matrix, rhs: Vector) -> List[float]:
    """前向代入解 ``L x = rhs``（L 为下三角）。"""
    n = len(lower)
    x = [0.0] * n
    for i in range(n):
        value = rhs[i] - sum(lower[i][j] * x[j] for j in range(i))
        x[i] = value / lower[i][i]
    return x


def invert_spd(a: Matrix) -> List[List[float]]:
    """对称正定求逆：``A^{-1}``，经一次 Cholesky 逐列回代，结果对称化。"""
    lower = cholesky(a)
    n = len(a)
    columns: List[List[float]] = []
    for col in range(n):
        # 解 L y = e_col，再解 L^T x = y
        eye = [1.0 if i == col else 0.0 for i in range(n)]
        y = solve_triangular(lower, eye)
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            value = y[i] - sum(lower[j][i] * x[j] for j in range(i + 1, n))
            x[i] = value / lower[i][i]
        columns.append(x)
    inv = _zero(n)
    for i in range(n):
        for j in range(n):
            inv[i][j] = columns[j][i]
    # 对称化消除舍入非对称
    for i in range(n):
        for j in range(i):
            mean = 0.5 * (inv[i][j] + inv[j][i])
            inv[i][j] = inv[j][i] = mean
    return inv


def quadratic(a: Matrix, x: Vector) -> float:
    """``x^T A x``。"""
    return sum(
        xi * sum(aij * xj for aij, xj in zip(row, x))
        for xi, row in zip(x, a)
    )


def matvec(a: Matrix, x: Vector) -> List[float]:
    return [sum(aij * xj for aij, xj in zip(row, x)) for row in a]


def jacobi_eigh(
    a: Matrix,
    tol: float = 1e-13,
    max_sweeps: int = 200,
) -> Tuple[List[float], List[List[float]]]:
    """对称矩阵 Jacobi 特征分解，返回 ``(升序特征值, 列向量为特征向量的正交阵)``。

    用于展示后验协方差的主方向（信息最弱/最强方向），维度一般较小。
    """
    check_symmetric(a, "Jacobi 输入")
    n = len(a)
    work = copy_matrix(a)
    vecs = identity(n)
    scale = max(
        (abs(work[i][j]) for i in range(n) for j in range(n)),
        default=1.0,
    )
    eps = tol * max(1.0, scale)
    for _ in range(max_sweeps):
        off = math.sqrt(
            sum(
                work[i][j] * work[i][j]
                for i in range(n)
                for j in range(i + 1, n)
            )
        )
        if off <= eps:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(work[p][q]) <= eps:
                    continue
                tau = (work[q][q] - work[p][p]) / (2.0 * work[p][q])
                t = (1.0 if tau >= 0 else -1.0) / (
                    abs(tau) + math.sqrt(1.0 + tau * tau)
                )
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                # 旋转 A
                for k in range(n):
                    akp, akq = work[k][p], work[k][q]
                    work[k][p] = c * akp - s * akq
                    work[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, apq = work[p][k], work[q][k]
                    work[p][k] = c * apk - s * apq
                    work[q][k] = s * apk + c * apq
                # 旋转特征向量
                for k in range(n):
                    vkp, vkq = vecs[k][p], vecs[k][q]
                    vecs[k][p] = c * vkp - s * vkq
                    vecs[k][q] = s * vkp + c * vkq
    eigenvalues = [work[i][i] for i in range(n)]
    eigenvectors = [[vecs[i][j] for j in range(n)] for i in range(n)]
    order = sorted(range(n), key=lambda i: eigenvalues[i])
    sorted_values = [eigenvalues[i] for i in order]
    sorted_vectors = [
        [eigenvectors[i][j] for j in order] for i in range(n)
    ]
    return sorted_values, sorted_vectors


def logdet(a: Matrix) -> float:
    """便捷封装：直接对对称正定矩阵求 ``log det``。"""
    return logdet_from_chol(cholesky(a))
