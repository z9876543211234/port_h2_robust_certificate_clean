# Port H2 Robust Certificate Clean

本仓库依据项目最终实施规格，从数学定义独立实现港口电—氢—LOHC—AGV
两阶段鲁棒优化及正式证书流程。运行时不导入、动态加载或调用任何历史模型。

## 边界

- 物理优化只覆盖一个调度日：正式配置仅允许 24×1 h 或 96×0.25 h。
- 风电和整数船舶延误由两个独立构建程序生成不可变
  `UncertaintyBundle`；物理模型只读取 Bundle 输出。
- 船舶构建器使用前一日、当前日、后一日的三日轴，但只向模型投影当前日。
- 风—氢分流固定；无燃料电池、无日前 LOHC 外运、无日前 AGV 工作分配。
- 固定场景 recourse、master、Phase-I、dual、adversary 和残差由同一线性 IR
  编译。
- 正式证书只接受 `GRB.OPTIMAL`；任何限时或其他状态均失败关闭。

## 当前正式快照（2026-07-14）

GitHub 快照的权威算例位于
`experiments/c1_confirmed_main_budget_20260714/`，采用修正后的四键互斥分区。
正式参数为：60 辆 AGV、18 台充电机、充电/LOHC 调整上限
900/1,200 kW、对应调整成本 0.3/0.3 元/kWh、电网偏差成本
0.4 元/kWh、充电数量爬坡 4 辆/时段、风电预算 12、整数延误船舶预算
2 艘以及最大延误 4 个时段。弃风日前/日内/偏差成本为
0.1/0.1/0.125 元/kWh。

权威输出目录为：

- `C1_ProposedRobustMain_refined_partition`
- `C2_NoHydrogenRobust_refined_partition`
- `C3_WorkCapacityCapsRobust_refined_partition`
- `C4_DeterministicMain_refined_partition`

四个算例的 master、Phase-I 和成本 adversary 均要求且达到
`GRB.OPTIMAL`，未接受限时解。汇总、输入哈希和证书说明见
`C1_C4_FOUR_KEY_RESULTS.md` 与 `FINAL_RESULT.md`。当前论文图位于
`paper_figures/formal_c1_c4_four_key_20260714/`，补充的日前—日内、LOHC
库存和 AGV 分配图位于 `paper_figures/day_intraday_operation_20260715/`。
当前快照在 2026-07-15 完整回归验证为 `110 passed in 135.40s`。

## Python 环境

要求 Python 3.12+。当前已验证解释器为：

```text
/opt/anaconda3/bin/python3.13
```

运行测试：

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python3.13 -m pytest -q -p no:cacheprovider
```

正式输入会复制到本仓库自身的 `data/`，并在 manifest 中记录原始文件、规范化
载荷和生成脚本的 SHA-256。运行期不得读取旧项目路径。

## 基础输入模板状态

旧项目 96 时段确定性参数和风电序列已迁移到 `data/`。船舶名义到港则由
`src/port_h2_uncertainty_builders/ship_delay/nhpp_nominal.py` 离线生成：沿用旧项目
记录的周期 NHPP 强度形状和日均期望 11.52 艘，以固定种子 `20260713` 对三个
相邻日分别抽取独立 Poisson 整数计数。前日、当日、次日实现总数依次为
15、15、9 艘。96 时段结果直接冻结；24 时段结果由同一次抽样每四个时段求和，
不重新抽样。

NHPP 的小数期望只保存在审计 CSV 中，不进入模型名义量。模型只读取冻结的
非负整数船数，并只对当日逐艘船建立正整数延误选择；不允许连续等效到港压力、
提前到港、跨日取模或求解期间重新抽样。运行
`python tools/migrate_authoritative_inputs.py` 可按同一固定种子完整复现数据及 hash。

基础模板中的以下值仍保持 `null`，正式加载器会阻断直接运行，直到人工填写。
上述权威正式快照使用 `experiments/c1_confirmed_main_budget_20260714/inputs/`
中的冻结配置，不受这些模板占位符影响：

- `spill_day_ahead_per_kwh`、`spill_real_time_per_kwh`、
  `spill_deviation_per_kwh`；
- `max_delay_steps`、`delayed_ship_budget`；
- 24 时段 H₂ 整数登陆延迟及其历史登陆量（旧值是 1 个 15 分钟步，无法无损
  映射为整数小时步）。

`total_delay_step_budget: null` 是规格允许的“禁用可选预算”，不是阻断项。

检查当前阻断项：

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python3.13 -m runners.validate_case \
  C1 --profile quarter_hour_96 --output /tmp/port-h2-validate
```

正式值填妥后运行单个案例：

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python3.13 -m runners.run_case \
  C1 --profile quarter_hour_96 --output results/C1
```

命令行 `--set dotted.path=<json-value>` 只用于受控验证，不会写回正式输入文件。
