# GPT-6.1 Sol 五任务对照（220 控制步）

已增加 `configs/gpt61_sol_medium_220.json` 和 `scripts/run_gpt_five.sh`。模型为 `gpt-6.1-sol`，推理强度为用户确认的 `medium`。本次只做离线实现、验证和估算，没有调用付费模型，也没有执行真实 LIBERO 回合。

## 实验协议

| 项目 | GPT 对照 |
|---|---|
| API | 官方 `https://api.openai.com/v1/responses`，Standard（`service_tier=default`） |
| 模型 / 推理 | `gpt-6.1-sol` / `medium`；别名不等于冻结权重 |
| 任务 | 下面固定的 5 项，每项 1 个回合 |
| 初始条件 | 官方 init_state=0、seed=0、task_order_index=0 |
| 控制预算 | **每项最多 220 个实际执行的 7D 动作**；10 个稳定步骤另计 |
| 动作块 | 一次输出 1–8 个动作；保留夹爪反向、抓取阶段切换处的重新观测 |
| 图像 / 历史 | 两路 256×256 PNG，最近 2 次视觉历史，最近 4 步动作和自身状态 |
| 几何反馈 | 沿用已跑五任务的 grip_site 坐标、指垫、相机几何和同步观测 |
| 单次输出上限 | 16,384 token，**包含 reasoning**；截断不执行、不自动修复 |
| 请求预算 | 每项正式回合最多 220 次 HTTP 尝试；另有 1 次付费 API 检查 |
| 重试 / timeout | 默认 0 次自动重试；socket timeout 600 秒，超时不代表服务端未计费 |
| 缓存 | 显式 implicit / 30m；缓存写入和缓存读取分别计价 |
| 状态 | 每个回合使用独立短历史；不使用 `previous_response_id`，不累积全部对话 |
| 输出判定 | JSON mode + 原有严格动作校验；拒绝、截断、越界和多余字段均不执行 |

这里的 220 步不是 220 次长动作块。若每次执行 8 个动作，跑满 220 步只需 28 次决策；若每次仅执行 1 个动作，则需 220 次。每项再加 1 次不执行动作的 API 检查，5 项的默认请求总上限为 **1,105 次**。标定不调用 API。

| 计划索引 | suite / task_id | 任务 |
|---|---|---|
| 0 | libero_spatial / 0 | 将黑碗放到盘子上 |
| 1 | libero_object / 0 | 将 alphabet soup 罐放进篮子 |
| 2 | libero_goal / 0 | 打开柜子的中层抽屉 |
| 3 | libero_goal / 5 | 将盘子推到炉子前面 |
| 4 | libero_10 / 5 | 将书放进收纳盒后格 |

名称和编号在 `ds_libero/five_tasks.py` 固定，并在仿真创建时核对实际任务名。已有 DeepSeek 五任务允许 1024 步，不能直接把其最终成功率当作同预算的 GPT 对照。应分别标注 1024 / 220 步；严格比较需要新建 DeepSeek 220 步计划。旧轨迹的前 220 步只能作为描述性参考，因为旧提示词告诉模型的预算仍是 1024。

## 密钥

读取顺序：`OPENAI_API_KEY` 环境变量，然后仓库根目录 `gpt_API.txt`（UTF-8，可带 BOM）。文件与 `ds_API.txt` 同目录，**只填密钥本身**。OpenAI 不会回退读取 DeepSeek 密钥，密钥也不写入请求快照、日志或计划。

本次检查时 `gpt_API.txt` 不存在，因此已创建空文件；需要在本机或实际运行节点自行填入真实密钥。本地 `.gitignore` 已包含两个文件，本次保留了用户原有修改。密钥文件不随 Git 同步，也不包含在代码交付包中；远程节点需要单独配置。

可先做不联网的启动检查：

```bash
SIM=/mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh
bash "$SIM" -B -m ds_libero check-config --config configs/gpt61_sol_medium_220.json
```

## 运行

将本次代码同步到实际仿真节点后，在仓库根目录运行。保持已有 LIBERO 环境；不需要安装 OpenAI SDK。

远程已有仓库时，可在交互式 Bash 中执行下面整段命令。使用实验分支 `codex/direct-policy-v2`；更新只能快进，遇到未提交改动冲突或分支分叉会停止，不覆盖远程节点上的个人修改。密钥缺失时隐藏输入并仅保存到 Git 忽略的本地文件，已有密钥则直接复用。

```bash
(
set -euo pipefail
set +x
cd /mnt/cpfs/users/mrq/DS4.1F1017-4-LIBERO
git fetch origin codex/direct-policy-v2
git switch codex/direct-policy-v2
git merge --ff-only origin/codex/direct-policy-v2

export LIBERO_SIM_RUNNER=/mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh
if [[ -z ${OPENAI_API_KEY:-} && ! -s gpt_API.txt ]]; then
  read -r -s -p 'OpenAI API key: ' gpt_key_input
  printf '\n'
  (umask 077; printf '%s\n' "$gpt_key_input" > gpt_API.txt)
  unset gpt_key_input
fi
bash "$LIBERO_SIM_RUNNER" -B -m ds_libero check-config \
  --config configs/gpt61_sol_medium_220.json

OUT="outputs/gpt61-sol-medium-220-$(date +%Y%m%d-%H%M%S)"
printf 'Results: %s\n' "$OUT"
bash scripts/run_gpt_five.sh "$OUT"
)
```

该流程前台运行，请保持终端连接，或在已有 tmux 会话中执行。脚本会保存日志；最后查看刚打印的输出目录中的 `aggregate.json`、`episodes.csv` 和各回合 `campaign_summary.json`。修复 JSON mode 请求后共 **115 项离线测试通过**，另已通过 Python 3.10 和 Bash 语法检查。

```bash
# 一条命令：冻结新计划，然后按顺序运行 5 个任务（此命令会消耗 API 额度）
bash scripts/run_gpt_five.sh outputs/gpt61-sol-medium-220-001
```

每项依次运行标定、1 次 API 检查、正式回合，并汇总开销。输出目录必须是新目录。可以用环境变量 `LIBERO_SIM_RUNNER` 指定其他仿真启动脚本。

如希望先单独跑第 1 项校准费用和耗时：

```bash
SIM=/mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh
# 创建计划免费，不连接 API，也不加载仿真。
bash "$SIM" -B -m ds_libero plan-five \
  --config configs/gpt61_sol_medium_220.json --out outputs/gpt61-sol-medium-220-001
bash "$SIM" -u -B -m ds_libero run-plan-episode outputs/gpt61-sol-medium-220-001 --episode 0
# 然后只启动尚未开始的任务；已记录的模型结果不会重新付费运行。
bash "$SIM" -u -B -m ds_libero run-plan outputs/gpt61-sol-medium-220-001
```

网络或鉴权等基础设施错误会停止流水线，保留记录，不会自动重跑已花费的回合。若某回合留有未完成目录，先检查其日志；可用 `run-plan-episode --episode N` 明确选择后续尚未开始的项。代码或配置改变后需新建计划。

```bash
# 免费重新生成报告
bash "$SIM" -B -m ds_libero report-plan outputs/gpt61-sol-medium-220-001
```

逐次 `api_calls.json/csv`、逐回合 `summary.json`、各阶段 `campaign_summary.json`、五任务 `aggregate.json` 和 `episodes.csv` 都记录实际 token、费用与等待时间。GPT 费用标注 **USD**，DeepSeek 保留 **CNY**；不做隐含汇率换算。`known_cost_range` 是已知小计，存在未知 usage 时 `total_cost_range=null`。缓存写入量缺失、返回服务档位不符或模型不匹配时，不把相关请求算成免费。

## API 检查返回 HTTP 400：input 缺少 JSON 指令

首轮远程运行的错误为 `Response input messages must contain the word 'json' in some form to use 'text.format' of type 'json_object'.`。此前 JSON 输出指令仅存在于 Responses 的顶层 `instructions`，而服务端检查的是 `input` 消息，因而在 API 检查阶段拒绝请求，未执行控制动作。该结果属于请求错误，不计为模型任务失败。[官方 JSON mode 说明](https://developers.openai.com/api/docs/guides/structured-outputs#json-mode)要求在消息中明确指示输出 JSON。

修复在 GPT 的用户 `input` 末尾补充一句 JSON 输出要求；共享策略提示、观测、图像、动作格式及限制仍沿用原配置。`gpt-6.1-sol`、medium、220 控制步、16,384 输出 token 和默认不重试均未改变。后续 HTTP 错误会把脱敏的 `message/type/param/code` 写入事件日志及 `summary.json` 的 `reason`。

更新代码后使用上面的完整运行命令创建**新输出目录**，重新标定和 API 检查。旧失败目录保留供分析；它冻结了旧源码指纹，不能直接续跑，也不要用 `replay-request` 重发其中保存的旧请求。新计划会先检查 API，成功后才开始正式控制。此次修复在离线测试中复现并消除了全部五项任务的同类 400；尚未用真实 API 或仿真验证。

## 费用与时间估算

2026-11-02 核对的 [OpenAI 官方价格](https://developers.openai.com/api/docs/pricing)：`gpt-6.1-sol` Standard 每百万 token，普通输入 **$2**、缓存读取 **$0.10**、30m 缓存写入 **$2.50**、输出 **$10**。Reasoning 是输出 token 的子集，不再重复相加；见 [模型说明](https://developers.openai.com/api/docs/models/gpt-6.1-sol) 和 [缓存计价说明](https://developers.openai.com/api/docs/guides/prompt-caching)。这里计算额度的标价消耗，实际付款或赠送余额抵扣以账户账单为准。

使用已有 `five-tasks-geometry8-1024-001` 各任务前 220 步内实际执行的动作分组，推算跑满 220 步需要的正式决策数分别为 **41、46、50、35、40** 次，加上 5 次 API 检查，共约 **217 次**。其中提前失败的任务按已有动作分组外推，没有当作完整 220 步实测。GPT 可能选择不同块长，所以该数量也只是估算。

| 情景 | 每次输入（含图像） | 每次输出（含推理） | 假设每次响应 | 五项总费用 | 五项串行总时间 |
|---|---:|---:|---:|---:|---:|
| 较轻 | 8,000 token | 2,000 token | 15 秒 | $7.81–8.68 | 约 1.1 小时 |
| 中间 | 10,000 token | 4,000 token | 30 秒 | $13.02–14.11 | 约 2.0 小时 |
| 较重 | 14,000 token | 8,000 token | 60 秒 | $23.44–24.95 | 约 3.8 小时 |

时间另加了全部任务合计 10 分钟的初始化、标定、仿真、日志和回放余量。输入区间按全部普通输入至全部 30m 缓存写入计价，不预支缓存命中节省。**GPT 的输出长度与速度均未实测，上表是规划情景，不是置信区间或保证。** 可以先按约 **$15、2 小时**安排，预留 **$30、4 小时**；首项运行后用实际 usage/latency 重估。

极端情景：五项都用满 1,105 次请求，每次输出打满 16,384 token，且假设每次 14K 输入都按 30m 缓存写入计费，约 **$219.72**。这不是硬金额上限：输入长度没有美元预算限制。若每次仍耗时 30–60 秒，串行约 9.4–18.6 小时；真实耗时也可能更长。220 步本身不能保证只花十几美元。

离线重算命令（不调用 API）：

```bash
python -m ds_libero estimate-gpt \
  --baseline '部分结果26.11.02/five-tasks-geometry8-1024-001'
# 无历史文件时仍可运行：使用平均每次执行 5 步的明确假设。
python -m ds_libero estimate-gpt
```

## 验证与实现来源

本地当前 checkout 尚无五任务和修正后的几何协议。本次从已有运行目录恢复了对应源码快照，并在复制前校验整个快照的 SHA256：`bc076b0c800abb2b5cd38a99d448b8f65a45b4119e7102e18293c74f679649ef`。原运行记录的提交为 `cb0c2620faa3ee717f8f1e7ee5d14a1a83e2d490`。这样 GPT 与已有五任务采用相同的观测、提示词和动作语义；恢复过程未修改历史结果。

离线测试覆盖 Responses 请求与解析、相同观测/图像、220 步执行边界、无重试、凭证隔离、缓存写入和美元计价、失败请求未知费用、冻结五任务、已有付费回合不重跑以及离线重放。已有 DeepSeek 测试一并通过。真实仿真与 GPT 账户权限仍由远程运行时的标定和 API 检查验证。
