# 受约束测量方案的增量信息设计

本项目保存虚构科学工程分析的合成输入与Python读取工具。当前工具仅解析输入并显示类型和条数，不执行参数拟合、统计分析、方程求解、优化或模型训练。所有仪器、实验、研究与专家身份均为虚构。

## 数据口径

候选测量row为参数特征向量，noise_variance必须为正；prior_covariance为参数先验协方差。cost和组下限约束均是配置给定，尚未有测量响应值。

fixtures保存原始输入，src/design_reader.py是当前读取器，material-index.json为文件内容摘要。研究计算输出可写到独立.workspace-output目录，输入不应被覆盖。

## 本地运行

Python 3.11及以上；已有读取工具仅依赖标准库，无第三方安装步骤。进入项目根目录执行：

    python -m src.design_reader fixtures/candidates.json

编译检查：

    python -m compileall -q src tests

输入契约验证：

    python -m unittest discover -s tests -v

## 固定容器

    docker build -t informative-measurements:local .
    docker run --rm informative-measurements:local

容器按完整镜像摘要固定，默认运行读取测试，不访问外部实验系统。
