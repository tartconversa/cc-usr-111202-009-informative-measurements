# 受约束测量方案的增量信息设计

本项目保存虚构科学工程分析的合成输入，以及在**明确先验协方差与独立噪声**
下比较候选线性测量的 Python 工具。所有仪器、实验、研究与专家身份均为虚构；
工具只做先验下的设计计算，**不执行任何设备操作，也不读取观测值**。

## 设计口径

候选测量为线性形式 ``y = a^T theta + e``，其中 ``e`` 独立、噪声方差为
``sigma^2``。给定先验协方差 ``Sigma0``（必须对称正定），选中集合 ``S`` 的
精度矩阵与信息增益为

```
P(S) = Sigma0^{-1} + sum_{j in S} a_j a_j^T / sigma_j^2
IG(S) = 0.5 * (log det P(S) - log det Sigma0^{-1})
```

这是 D-最优实验设计的信息增益（nats；除以 ``ln 2`` 得以 2 为底的对数单位）。
观测值尚未到达，选择只依赖特征行、噪声、成本与约束；候选记录若携带响应类
字段（``observation``/``response``/``y`` 等）会被直接拒绝。

### 硬约束

* 总成本不超过 ``budget``；
* 每个仪器组的测量数满足 ``group_min`` / ``group_max``；
* 互斥标签：同一标签至多被一个选中测量占用；
* 同一 ``position`` 的选中数不超过 ``position_upper_bound``；
* 空方案只在满足全部组下限时可行，否则判不可行并给出原因。

### 两个基准数值（已在测试中核对）

二维单位先验、单位噪声：选 ``a=(1,0)`` 与 ``(0,1)`` 时增益为 ``log 2``，
后验协方差为 ``diag(0.5, 0.5)``；若改成两个同位置 ``(1,0)``，增益仅
``0.5 log 3``，且第二方向后验方差仍为 1（未被覆盖）。报告的方向方差表与
主方向分解直接呈现“多测一次同方向”和“覆盖新方向”的差别。

## 求解方法

| 方法 | 适用 | 保证 |
| --- | --- | --- |
| `exact` | 小规模 | 分支定界精确最优；受 `search_node_budget` 限制停止时返回当前可行解、可证明上界与 gap |
| `incremental` | 可扩展 | 可行性骨架（满足组下限）+ 懒评估增量贪心；启发式，给可行解与可证明上界、gap |

`log det` 经 Cholesky 计算，测量加入走 O(d²) 秩一 Cholesky 更新；后验协方差
走 Sherman-Morrison 秩一更新，两条路径在测试中交叉核对。一般基数/预算约束下
贪心**没有固定近似比**，工具不声称任何比率，只报告与合法上界的 gap。上界由
预算松弛、预算基数、互斥/位置分区三个可证明分量取最紧值得到。

### 未选候选的两类解释

* **约束不可加入**（`constraint_blocked`）：在最终方案状态下直接违反预算、
  组上限、位置上限或互斥，给出具体原因；
* **重新优化后的全局机会损失**（`reoptimized`）：约束上可加入，但强制包含
  该候选并重新优化后的最优值严格更低，报告机会损失（启发式下给损失区间）。

当下边际增益只作为描述量展示，**不能替代全局结论**。

## 本地运行

Python 3.11 及以上，仅依赖标准库，无第三方安装步骤。在项目根目录：

    # 小规模精确设计
    python -m src.cli fixtures/candidates.json fixtures/design.json --method exact
    # 可扩展增量搜索（候选文件内嵌设计字段时可省略第二个路径）
    python -m src.cli fixtures/candidates-large.json --method incremental
    # 两种方法并排比较
    python -m src.cli fixtures/candidates.json fixtures/design.json --method compare
    # 机器可读 JSON，并把报告写入独立输出目录（不覆盖输入）
    python -m src.cli fixtures/candidates.json fixtures/design.json --json \
        --output .workspace-output/small-design.json

原有读取器保持只负责读配置：

    python -m src.cli --help
    python -m src.design_reader fixtures/candidates.json

编译检查与全部测试（含数值基准、近共线、预算无解、不等噪声、并列固定键
次序、上界合法性枚举）：

    python -m compileall -q src tests
    python -m unittest discover -s tests -t . -v

在隔离模式（`-I`，忽略环境变量与当前目录注入）下运行时，测试可用
`python -I -m unittest discover -s tests -t .`；CLI 入口经 `python -m src.cli`
正常运行。

## 输出内容

文本/JSON 报告包含：求解状态与节点用量、固定键次序的选中方案、逐次选择的
信息收益与收益/成本比、后验协方差与精度矩阵、各参数坐标与候选特征方向的
先验/后验方差及相对下降、后验主方向（信息最弱方向）、噪声估计敏感性
（噪声标准差按配置倍数缩放后的增益、后验与重新优化方案变化），以及未选
候选的区分性解释。

## 固定容器

    docker build -t informative-measurements:local .
    docker run --rm informative-measurements:local

容器按完整镜像摘要固定，默认运行全部 unittest，不访问外部实验系统。
