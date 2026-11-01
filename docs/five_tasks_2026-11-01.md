# 8 步 chunk / 1024 控制步：五任务诊断

用户明确 1024 指每任务的**控制步数**，不是 API 推理次数。本轮回到 `pilot-v2-geometry-high-001` 的控制设置：系统提示词逐字相同，最多 8 条原生 7D 动作/组、high、请求省略 max_tokens、几何校验、双视角短历史、抓取反馈及阶段边界。仅扩大控制时限并增加任务，不增加脚本抓取、物体真值或动作重复规则。

## 任务清单

任务顺序 `task_order_index=0`；每项都用自己的官方 `init_state_id=0` 和 `seed=0`，各运行一次。ID 和完整任务名已对照 [固定 LIBERO 提交的任务表](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/benchmark/libero_suite_task_map.py) 及其 identity task order 核实，运行时再次检查名称。

| 批次序号 | suite / task_id | 任务 | 观察目的 |
|---:|---|---|---|
| 0 | libero_spatial / 0 | 原黑碗放盘子任务 | 与已有轨迹延续对照 |
| 1 | libero_object / 0 | alphabet soup 汤罐放入篮子 | 不同形状物体抓取与容器放置 |
| 2 | libero_goal / 0 | 打开柜子的中层抽屉 | 把手接触和关节操作 |
| 3 | libero_goal / 5 | 将盘子推到炉灶前方 | 无需先抓起的推动 |
| 4 | libero_10 / 5 | 书放入收纳架后格 | 书房场景、扁平物体与插放 |

这是五个固定任务的延长时限诊断，不是全套 LIBERO 分数，也不能从每任务一次试验估计稳定成功率。Goal 的两项共享套件布局，但操作目标不同。

## 时限、调用和费用口径

- 新基线配置：`configs/direct_v2_geometry_1024_high.json`。生成的五份配置使用独立 `diagnostic` scope；标准 benchmark 各套件时限保持原样。
- 20 Hz；每 chunk 最长 0.4 秒仿真；每任务最多 1024 步/51.2 秒仿真，成功立即停止。
- 底层 horizon 为 1035 = 10 步稳定 + 1024 控制步 + 1，避免 1000 步隐藏时限。
- 每正式回合 HTTP 尝试最多 1024 次，**包括重试**。API-check 最多另有 3 次。此额度不是计划调用数，也不是金额预算；全 8 步组跑满需 128 次有效决策，短组会更多，重试可能让 HTTP 预算先耗尽。
- high 和 max_tokens=null 保留。上轮的服务端默认 64K 截断仍可能发生；不会自动增加 token 上限或重复请求补救。`invalid_output` 结束当前任务、保留开销，再继续下一任务。
- 费用包含 reasoning（已属于 completion，不重复相加）、输入缓存命中/未命中及所有已记录尝试；未知费用不视为零。每任务的 reset 后墙钟、HTTP/decision latency、仿真时间分别保留。campaign 包含标定及 API-check；正式回合指标单列。
- 规模参考：若机械地按旧 geometry-high 轨迹的每步开销比例外推，五项全部跑满约 43 小时、人民币估算区间约 122–244 元。这只是旧单条轨迹的线性情景估算，新任务、短组、提前成功/截断均会改变结果，不是费用承诺或上限。

## 一次启动五任务（远程执行，会调用 API）

```bash
cd /mnt/cpfs/users/mrq/DS4.1F1017-4-LIBERO
git switch codex/direct-policy-v2
git pull --ff-only origin codex/direct-policy-v2
bash scripts/run_five_tasks.sh outputs/five-tasks-geometry8-1024-001
```

沿用项目 `ds_API.txt` 或环境变量，原 CPU/OSMesa launcher 不变，无需安装新模型或修改依赖。

脚本先离线生成固定清单和五份配置，然后串行执行。每项独立标定、API-check、重置官方初态、正式控制；每项结束后刷新总报告。退出码 0 只表示批次已处理，不代表五任务全部成功，应看 aggregate 的各项状态。

已记录 success / failure / invalid_output / budget_exhausted 的正式回合，以及 API-check invalid_output，重复启动时跳过，不自动重新付费。API-check 失败不算正式任务失败，但其费用计入对应任务 campaign。budget_exhausted 保持未评分，不伪装成任务失败。

标定、鉴权、网络、仿真等基础设施错误会停止批次并保留报告。Ctrl+C 可中止；中断回合不自动恢复，也不自动重新付费。排查后可以显式运行其他尚未开始的序号，不需改写失败记录。代码变化会使旧清单拒绝执行，不能把不同策略结果静默混入同一批次。

## 分任务运行与重新汇总

只准备清单（无 API 调用）：

```bash
SIM=/mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh
bash "$SIM" -B -m ds_libero plan-five \
  --config configs/direct_v2_geometry_1024_high.json \
  --out outputs/five-tasks-geometry8-1024-001
```

运行单个尚未开始的任务，例如序号 2；序号范围 0–4：

```bash
bash "$SIM" -u -B -m ds_libero run-plan-episode \
  outputs/five-tasks-geometry8-1024-001 --episode 2
bash "$SIM" -B -m ds_libero report-plan outputs/five-tasks-geometry8-1024-001
```

`run-plan` 可继续尚未开始的任务。已存在但未完整结束的目录会被保留并提示排查；单任务命令也拒绝覆盖输出目录。

## 回传结果

```text
plan.json
aggregate.json                    五任务状态、覆盖率、逐任务费用/latency
episodes.csv                      每任务一行，可直接比较
configs/episode-0000.json ... 0004.json
runs/episode-0000/ ... episode-0004/
  campaign_summary.json           本任务所有阶段费用
  calibration/summary.json
  episode/summary.json
  episode/events.jsonl
  episode/api_calls.json          每个付费尝试用量与费用
  episode/api_calls.csv
  episode/replay.gif
  episode/responses/              若出现截断，保留原响应供分析
```

总报告按 `suite:task_id` 分组，避免三个不同套件的 task 0 混合。保留未开始、中断、基础设施错误与模型输出失败的不同状态；缺失任务不会从分母中消失。知识库另行在本地维护，不包含在远程推送中。
