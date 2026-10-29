# DS4.1F-1017 × LIBERO：当前证据与下一轮实验

> 2026-10-29 更新：下文保留 10 月 28 日的实验设计记录。16K high 精确重放也发生 length 截断后，用户决定保持 high、取消客户端 token 限制。当前 v2 high 配置已改为 null / 1800 秒；最新命令和开销、latency 定义见 [新的运行说明](cost_latency_2026-10-29.md)。原 v1 / 16K / low 配置供历史对照。

## 已有证据

本次核对了同项目旧会话、`D:/Code_Projects/dsh1017test/测试json/summary.json`、`第一次执行结果/events.jsonl`、`result.json`、`summary.json` 和两份原始响应。原始文件保留原位，不把分析文本当作环境真值。

| 检查 | 已确认结果 |
|---|---|
| 仿真环境 | CPU / OSMesa，256×256 双视角；12 个正负方向探针和夹爪开合标定通过 |
| API 与执行 | 第一条有效 7 维动作被执行，随后两路观测更新 |
| 第一次请求 | 28.719 s；5609 completion tokens，其中 5587 reasoning tokens；`stop` |
| 第二次请求 | 45.823 s；8192 completion tokens 全部是 reasoning；最终 content 为空，`length` |
| 回合终止 | `invalid_output`；只执行了 1 / 220 个控制步，非网络超时、非仿真死锁 |
| 第一步运动 | 目标增量 X +25 mm / Y −25 mm，实际约 +2.353 / −2.723 / −0.339 mm，总位移 3.615 mm |
| 评分边界 | 原协议下模型输出失败计该回合失败；尚无完整抓取结果，不能计算 LIBERO 总体成功率 |

原记录结束于北京时间 2026-10-21 21:44:36。`RESULT:` 已在关闭环境、保存结果与生成回放之后打印；后续 `tail -f` 等待不等于实验还在运行。先前缺少 API key 的启动在调用前已退出。

## 需要解决的问题

1. **输出截断已证实。** `high + 8192` 在第二次决策没有留下最终动作。提高预算是待测假设，不能保证成功。
2. **空间对应信息不足。** 原输入没有相机标定、历史图像；返回文本反复猜测图像方向与坐标轴，并误猜图像尺寸。不能仅据此断言它认错了目标碗。
3. **单步动作效率低。** 一次几十秒的请求只推进 0.05 s 仿真；目标偏移和实际位移差异不证明缩放错误。需要测试动作序列，同时保留每步成功检测。
4. **诊断可追溯性不足。** 旧版保存请求哈希，没有请求体；重建第二次请求还需要远程原目录的 manifest、prompt 和 frame 00001 的双视角图片。本机收到的第一次执行结果没有这些文件。
5. **比较协议需固定。** 模型别名会更新；动作接口、历史、预算、几何信息同时改变时无法单独归因。基础设施未完成回合须与模型失败分开统计。

## 参考实验与本实验的对应关系

[Astranomy](https://research.nvidia.com/labs/srl/projects/astranomy/) 将 Direct Policy、Agentic Policy 和 Orchestrated Policy 分开。其 Direct 配置使用绝对 EEF 目标、动作序列、短观测历史及几何信息；由此借鉴输入和执行接口的消融维度，不搬用其成功率。

RoboDojo 网页在本次文本抓取中未返回正文，交叉核对了其[同名原始报告](https://arxiv.org/html/2609.24170v1)及[官方执行适配器](https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/GPT_6_Astra_Direct_EEF/README.md)：LLM 提供 EEF 目标，由非学习控制转换模块执行。这与“每次只发一个归一化增量”不是同一种接口。报告中的 DeepSeek-Flash 也不能自动等同于本次预期的 1017 版本。

本轮 v2 **仍用 LIBERO 原生增量 OSC_POSE**，模型选择所有可执行动作；没有 VLA、物体真值、示范、任务专用脚本或外部 agent 工具。每次 API 仅接收固定提示词、当前观测和有界历史，属于 direct policy 实验。它并非两份参考报告的完全复现。绝对 EEF/TCP 控制及抓点偏移需要独立控制器标定，留作后续动作表示对照，不在本轮悄悄替换。

## 已实现的实验变体

下表所有现成 pilot 配置仍固定黑碗 task=0、init_state=0、seed=0、20 Hz、220 个控制步；不自动扩展到付费全基准。只有显式生成 benchmark 清单才可改变任务与初始状态。

| 配置 | 相对上一行的改动 | 目的 |
|---|---|---|
| `first_episode.json` | 原版 high / 8192 / 单步 | 保存旧基线，原配置指纹不变 |
| `v1_high_16k.json` | 只把 max_tokens 改为 16384 | 排查截断 |
| `v2_base_high.json` | v2 动作列表协议，仍仅一条动作 | 测量提示词与 JSON 格式变化影响 |
| `v2_geometry_high.json` | 加入动态相机内外参及 EEF 轴投影 | 减少世界坐标与图像方向歧义 |
| `v2_history_high.json` | 加入最多两次历史决策时的双视角图像与状态 | 提供可对照的视觉运动反馈 |
| `direct_v2_high.json` | 每次允许 1–8 个动作 | 减少请求次数，测试短时开环代价 |
| `direct_v2_low.json` | 只把 high 改为 low | 推理强度对照 |

v2 的 `actions` 中每项只执行一次 `env.step()`，每一步的增量相对该步实际末端姿态；重复增量不是保持同一目标。整段先验证，任何非法项都会拒绝整段；最后一段不得超过剩余步数。成功或异常终止会立即丢弃余下动作。最大 chunk=8 对应 0.4 s 仿真开环，模型可选择更短序列。

相机 `K`、`T_world_from_camera` 使用 robosuite 1.4.1 的光学坐标约定（X 右、Y 下、Z 前）；原图只翻转一次。轴投影仅使用机器人末端和相机参数，没有物体定位。几何数学已离线测试，**远程真实相机图像与几何一致性仍需本轮标定核验**。参考：[robosuite 相机实现](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/utils/camera_utils.py)。

DeepSeek 文档确认 `medium` 会映射到 `high`，所以真正降低强度的对照用 `low`。两者保持 thinking enabled 与 16384 上限。[官方说明](https://api-docs.deepseek.com/guides/thinking_mode/)

## 远程执行顺序

代码更新后在仓库根目录运行，继续使用已有环境。API key 仅在远程当前 shell 设置，不放进聊天、脚本、JSON 或 Git。

```bash
cd /mnt/cpfs/users/mrq/DS4.1F1017-4-LIBERO
SIM=/mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh
read -r -s -p 'DeepSeek API key: ' DEEPSEEK_API_KEY
printf '\n'
export DEEPSEEK_API_KEY
```

第一步先验证失败请求能否被精确重建（不调用 API，不启动仿真）：

```bash
bash "$SIM" -u -B -m ds_libero replay-request outputs/episode-001 \
  --step 1 --max-tokens 16384 --dry-run --out outputs/replay-step1-preflight-001
```

只有请求 SHA256 与原日志匹配才允许重放。缺文件或哈希不符会报错，不能用初始图像代替失败帧。检查通过后执行预算诊断（通常一次付费请求；网络错误最多再试两次）：

```bash
bash "$SIM" -u -B -m ds_libero replay-request outputs/episode-001 \
  --step 1 --max-tokens 16384 --out outputs/replay-step1-high16k-001
```

`pass` 仅表示该观测下动作输出完整合法，没有执行动作，也不计任务成功。它不作为正式回合 API gate。若仍然 `length`，保留结果；可另建目录加 `--reasoning-effort low` 做对照，禁止把后一次响应补进旧回合。

第二步跑候选 v2：

```bash
bash scripts/run_pilot.sh configs/direct_v2_high.json outputs/pilot-v2-high-001
```

脚本前台依次执行标定、API 检查、正式单回合；仅前项明确 `pass` 才继续。即使 API 检查的 invalid_output 退出码为 0，也不会误启动回合。每段有独立 console log，终端会正常返回，不自动启动永久 `tail -f`。旧 gate 的源码指纹不同，需要按当前代码重新标定。

如果先做单变量诊断回合，将配置替换为 `configs/v1_high_16k.json`，输出目录换为 `outputs/pilot-v1-high16k-001`。low 对照使用 `configs/direct_v2_low.json` 和新的目录。当前没有自动遍历所有配置的脚本，以免一次启动大量 API 调用。

需要断开终端时，建议在已有 tmux 会话内运行前台脚本；不要仅凭终端静止判断挂起。查看 `api_started`、`api_attempt`、`step` 事件及最终 `RESULT:`。输出目录已存在会拒绝覆盖。

## 结果与下一阶段

每个新回合保存实际请求 JSON（内含输入图像、无 API key）、原始响应、finish_reason、system_fingerprint、逐步动作、双视角回放、source snapshot。报告区分请求次数、决策次数和执行动作数，记录截断次数、动作序列长度、token 用量与推理延迟。`system_fingerprint` 是提供方返回标识，不证明固定权重。

请先提供 `outputs/replay-step1-high16k-001/summary.json` 和 `outputs/pilot-v2-high-001/episode/summary.json`；若流水线提前停止，提供该阶段的 summary 与 console log。分析控制行为时再取相应 events、response、frames，避免只看有效 JSON 就判断视觉控制成功。

通过单回合诊断后，下一阶段再冻结协议，分离调参集和测评集，预登记 task / official init-state / seed 清单，按相同初始条件做配对比较。统计环境二值成功率、每任务样本数、置信区间、输出截断率和基础设施未完成率，同时报告仿真时间、墙钟时间、token 与回放。不能只报告成功回合延迟，也不能把两个不同基准的 score 直接比较。本轮不产生新的真实模型成绩。

## 多回合清单与汇总工具

已实现 `plan`、`run-plan-episode`、`report-plan`，用于上述下一阶段。生成清单及汇总没有 API 调用；执行命令一次只接受一个明确的回合索引，依次通过标定、API 检查与正式回合，没有默认全量循环。

以下示例生成 4 个探索回合（不是完整基准）；先完成前述单回合诊断再执行它们：

```bash
bash "$SIM" -B -m ds_libero plan \
  --config configs/direct_v2_high.json --suite libero_spatial \
  --task-ids 0 1 --init-state-ids 1 2 --seed 0 \
  --out outputs/spatial-pilot-plan-001

# 实际付费执行：这里只运行清单中的第 0 个回合。
bash "$SIM" -u -B -m ds_libero run-plan-episode \
  outputs/spatial-pilot-plan-001 --episode 0

# 任意阶段均可汇总，未执行的回合也会明确显示。
bash "$SIM" -B -m ds_libero report-plan outputs/spatial-pilot-plan-001
```

清单保存每个回合的完整配置、配置哈希、代码哈希和清单哈希，拒绝重复身份、配置漂移、混用 high/low 或混用不同仿真运行时。代码变化后须新建清单；已有结果目录不覆盖，失败回合不会自动重复直到成功。初始状态索引在创建清单时按 0–49 验证，在真实环境加载时还会核对对应任务实际拥有的官方初始状态数量。

支持 Spatial / Object / Goal / LIBERO-10，步数上限分别 220 / 280 / 300 / 520，参考 [OpenVLA 评测实现](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py)。这只借用回合时限，不宣称与其输入、策略或成绩可直接等价比较。

`aggregate.json` 与 `episodes.csv` 记录：

- 预定、已计分、成功、未完成的回合数，完成覆盖率和状态分类；
- 已计分回合 SR、逐任务 SR 与 Wilson 95% 区间，按预定总数计算的成功率上下界；
- 仅在全部回合可计分时给出完整任务平均 SR；
- 输出有效率、截断率、所有回合的请求延迟与墙钟时间、已报告 token；
- 发出请求后被硬中断且未留下响应的尝试仍计入请求数，usage 标为未知，而不是零费用。

`invalid_output` 仍计模型失败；鉴权、网络、预算、仿真异常、人工中断及未执行回合保留为未计分。API preflight 的失败单独显示；汇总中的运行成本仅包含正式回合，前置检查费用需单独看其 summary。Wilson 区间描述已计分的二值试验，并不能证明对未测任务的泛化能力。保留 `plan.json` 原件用于记录测评前冻结的样本，正式结果发布前应检查服务标识变化与未完成样本，不能只挑成功回放。
