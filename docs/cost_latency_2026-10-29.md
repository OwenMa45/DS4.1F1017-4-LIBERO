# High 推理、API 开销与任务 latency

本次按用户要求保持 `reasoning_effort=high`，取消实验客户端的 token 上限。`direct_v2_high.json`、`v2_base_high.json`、`v2_geometry_high.json`、`v2_history_high.json` 均设 `max_tokens: null`；实际 HTTP 请求完全省略 `max_tokens`，不传字符串或 JSON null。历史 `first_episode.json`、`v1_high_16k.json`、`direct_v2_low.json` 保留原设置。

[DeepSeek Chat API 文档](https://api-docs.deepseek.com/api/create-chat-completion/)在 2026-10-29 说明：省略该参数时 thinking / high 默认输出上限为 64K，最大可指定 384K。因而这里的“取消上限”准确含义是客户端不指定预算，服务端限制仍然存在；若再次返回 length，仍保存响应并作为 invalid_output，不自动修复或增加预算重试。

当前 high 配置和 replay 命令的 HTTP socket timeout 为 1800 秒，以容纳较长的推理等待。这不是推理预算，也不是严格的整次请求墙钟截止时间：底层 urllib 的 timeout 约束阻塞 socket 操作。API 次数、控制步数及最多两次网络重试的既有规则继续生效。重试和推理等待均不推进仿真。

## 下一次远程执行

在已有 `DEEPSEEK_API_KEY` 的远程终端中更新分支：

```bash
cd /mnt/cpfs/users/mrq/DS4.1F1017-4-LIBERO
git switch codex/direct-policy-v2
git pull --ff-only origin codex/direct-policy-v2
```

先以同一失败输入验证 high / 省略 token 限制。此命令会调用付费 API，但不启动仿真、不执行动作；输出目录必须是新的。

```bash
bash /mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh -u -B \
  -m ds_libero replay-request outputs/episode-001 \
  --step 1 --reasoning-effort high --no-token-limit \
  --out outputs/replay-step1-high-default-001
```

默认已不指定 token 上限，`--no-token-limit` 用于让命令意图更清楚。可加 `--dry-run` 仅验证旧请求 SHA256 和保存修改后的请求。新 replay 报告也记录本次实际耗时与开销，`scored=false`，不能作为任务成功率样本。

有效输出验证后，可执行一个 v2 high 正式回合：

```bash
bash scripts/run_pilot.sh configs/direct_v2_high.json outputs/pilot-v2-high-default-001
```

脚本在前台依次执行标定、API 检查、正式回合；任何前置检查未通过都会停止。退出时汇总已经发生的开销，前置检查失败也会产生 `campaign_summary.json`。代码和配置已变化，因此旧标定/API 检查不能作为本轮 gate，脚本会重新执行。

## 每个任务的记录

| 文件 | 内容 |
|---|---|
| `episode/summary.json` | 正式回合 token、请求数、费用估算、实际任务耗时与 API 延迟分布 |
| `episode/api_calls.json`、`api_calls.csv` | 每次请求的任务身份、step、attempt/retry、开始/结束时间、响应 ID、模型/fingerprint、finish_reason、token 分类和费用估算 |
| `api-check/summary.json`、`api_calls.*` | 付费前置检查的独立统计，包括截断输出 |
| `campaign_summary.json` | 标定、API 检查、正式回合所有已记录阶段的开销总和 |
| `manifest.json` | 配置、源代码指纹及本次价格快照 |
| `events.jsonl`、`requests/`、`responses/` | 可重新计算报告的原始审计记录，不含 API key |

CSV 未知数值留空；JSON 对应值为 null。报告中的 `reported_usage` 是已知用量之和，必须结合 `usage_incomplete` 和 `usage_field_available_attempts` 阅读；例如未返回 reasoning 分类并不意味着 reasoning 为 0。

`api_cost` 字段包括：

- `known_cost_range_cny`：已获取完整计价字段的请求，按空闲／高峰价计算的合计区间。
- `total_cost_range_cny`：所有已记录请求都可计价时等于上述区间；存在未知请求时为 null。
- `priced_attempts`、`unknown_cost_attempts`、`incomplete`：费用统计覆盖情况。

按 2026-10-29 核对的[官方人民币价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)，Flash 每百万 token 的空闲／高峰价分别是：缓存命中输入 0.02 / 0.04 元，未命中输入 1 / 2 元，输出 4 / 8 元。公式为 `(cache_hit × 命中价 + cache_miss × 未命中价 + completion × 输出价) / 1,000,000`。Reasoning 已包含在 completion 中，不重复收费。

估算区间不是实际账单：高峰规则包含中国法定节假日，代码不猜测节假日或服务端跨时段计费方式；同时保留精确请求时间与 usage 供核账。价目按运行时快照冻结，重生成报告不会静默套用新价格；旧运行若没有价格快照，仍能报告 token，但费用标为未知。价格变更后需更新快照并建立新实验计划。

截断、非法输出即使没有执行动作，也计入已发生的 token/费用。超时、网络失败、进程中断可能已产生服务端费用而未返回 usage，不能当作零费用；重试是独立 HTTP 尝试，全部记录。

## Latency 的定义

| 字段 | 时间范围 |
|---|---|
| `task_execution_wall_seconds` | 正式回合 reset/settle 完成后到任务循环退出，包含初始观测保存、全部 API 等待、重试退避、仿真执行和日志保存 |
| `command_wall_seconds` | 该命令构建环境至关闭环境的实际总耗时；包含 reset，不含最后的报告/GIF 生成 |
| `control_loop_wall_seconds` | reset/settle 开始至任务循环结束 |
| `http_latency_seconds` | 每次 HTTP 尝试完整等待时间的 count/mean/p50/p95/max/total，包含已完成计时的失败请求 |
| `decision_latency_seconds` | 请求组装、所有尝试与退避、解析动作的逻辑决策耗时；不是首 token 延迟 |
| `retry_backoff_seconds` | 重试之间实际等待的合计 |
| `simulation_step_seconds` | 已完成 `env.step` 的实际墙钟合计 |
| `simulated_control_seconds` | 已执行模型动作数 / 20，与实际等待时间无关 |

这些指标有包含关系，不可全部相加。回放 GIF 仍按仿真时间播放，不能用 GIF 长度作为任务 latency。失败/中断回合记录的是运行至终止的耗时；只有 success 的耗时才是成功完成任务的时间。硬中断而缺少结束记录时，HTTP 统计标 `http_latency_incomplete=true`，任务完整耗时不凭文件时间补造。Replay / API 检查没有任务执行 latency，只报告其请求和命令耗时。

## 多任务汇总与恢复报告

已有冻结 plan 使用 `report-plan` 后，`aggregate.json` 的 `per_task` 将汇总每个任务的各初始状态：token 分类、API 估算开销、等待合计、实际任务耗时分布和有效样本数。`episodes.csv` 保留逐回合指标。`campaign_*` 字段额外包含前置检查开销，避免正式回合未启动时漏算花费。未执行样本和基础设施中断仍保留在 coverage 中。

```bash
# 硬中断后，基于已落盘记录恢复汇总，无 API 调用
bash /mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh -B \
  -m ds_libero report outputs/pilot-v2-high-default-001/episode --no-replay
bash /mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh -B \
  -m ds_libero report-campaign outputs/pilot-v2-high-default-001
# 使用自己的新 plan 目录；旧 plan 冻结了旧代码，不能继续混跑
# bash .../run_cpu.sh -B -m ds_libero report-plan outputs/your-new-plan
```
