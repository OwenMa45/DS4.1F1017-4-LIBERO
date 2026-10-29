# DeepSeek-V4.1-Flash-1017 × LIBERO

零样本、视觉反馈、单步 7 维动作控制实验。首版只运行 LIBERO Spatial 的黑碗任务一个回合，不将单回合结果包装为整体基准分数。

**2026-10-28 更新：首轮已跑通动作链路，但第二次请求耗尽 8192 token，回合以 invalid_output 结束。** 真实标定已通过；目前没有完整任务成功成绩。

已增加失败请求精确重放，以及相机几何、短视觉历史、1–8 步动作序列的 v2 direct-policy 实验。原 v1 配置和提示词保留。[问题总结与消融设计](docs/experiment_status_2026-10-28.md)。

**2026-10-29 更新：保持 high，取消客户端 token 上限，并按任务记录 API 开销和实际 latency。** 当前 v2 high 配置使用 `max_tokens: null`，请求省略该字段；HTTP socket timeout 延长至 1800 秒。API 仍有服务端默认限制，不表示无限输出。每个阶段新增 `api_calls.json/csv` 明细，任务流水线新增 `campaign_summary.json`，包含前置检查的费用。多任务报告按 task 汇总 token、人民币估算区间、API 等待与任务耗时。[当前运行命令与统计口径](docs/cost_latency_2026-10-29.md)。以下为保留的原 v1 使用说明；新实验请用当前命令。

重放命令现支持 `--prompt-api-key`：环境变量缺失时可在远程终端隐藏输入，本次使用后不保存。缺少密钥会在创建实验目录前退出，不计为模型决策或 API latency；已有失败目录请保留并换新目录重跑。

**首个 v2 完整控制回合：37 次合法输出、220 步，任务未成功。** 抓取诊断发现闭合后过早抬升/横移；新增 `configs/direct_v2_grasp_high.json`，补充指垫投影、夹爪及实际运动反馈，并在动作阶段切换处重新观测。原配置保留，尚待远程验证改善效果。[抓取诊断、控制改进与下一轮命令](docs/grasp_control_2026-10-29.md)。

**EEF 几何核对：旧观测的位置来自 grip_site，姿态来自 hand body，二者相差绕 Z 轴 -90°。** 新配置 `configs/direct_v2_geometry_high.json` 统一控制参考系，校验指垫与相机关系，完善几何提示词并生成独立标定投影图。high、不显式限制 token、逐任务开销及 latency 记录保留。[几何核对、验证范围与最新实验命令](docs/geometry_contract.md)。

## 1. 获取代码

在远程节点执行。实验仓库与已有仿真环境分开存放：

```bash
cd /mnt/cpfs/users/mrq
git clone https://github.com/OwenMa45/DS4.1F1017-4-LIBERO.git
cd /mnt/cpfs/users/mrq/DS4.1F1017-4-LIBERO
SIM=/mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh
```

已有 checkout 时在该仓库执行 `git pull --ff-only`。以下命令均从实验仓库根目录运行。无需 `pip install`、无需模型权重、无需激活或改动现有 `.venv`。不要用系统 Python 启动仿真。

## 2. 动作标定（无 API 费用）

```bash
bash "$SIM" -B -m ds_libero calibrate \
  --config configs/first_episode.json \
  --out outputs/calibration-001
```

检查真实 Git revision、依赖、CPU/OSMesa、控制器缩放和两路 256×256 图像。每个方向的标定均重置为官方初始状态，分别测试 ±X/±Y/±Z 平移和旋转，最后测试夹爪关闭与打开。图像与数值记录在输出目录；标定的微小位移只验证方向，不要求实际位移精确达到目标偏移。

末尾应输出 `RESULT: pass`。若失败，保留目录并提供 `summary.json`；不要跳过检查或修改成功阈值来强行通过。输出目录必须不存在，重跑时改用 `calibration-002` 等新名称。

## 3. 图像与动作 API 检查（会调用付费 API）

在远程交互式 Bash 中输入密钥，不会显示输入内容，也不把密钥写入命令参数或文件：

```bash
read -r -s -p 'DeepSeek API key: ' DEEPSEEK_API_KEY
printf '\n'
export DEEPSEEK_API_KEY

bash "$SIM" -B -m ds_libero api-check \
  --config configs/first_episode.json \
  --calibration outputs/calibration-001 \
  --out outputs/api-check-001
```

该命令重新加载官方初始状态，执行稳定步骤后将双相机 PNG 和自身状态发给官方 API，验证最终 JSON 动作。**返回动作不执行**。正常只需一次请求，网络类错误最多追加两次尝试。`pass` 证明图像请求和输出接口可用，不证明模型正确理解了图像。

## 4. 正式单回合

仅在前两项通过后执行：

```bash
bash "$SIM" -B -m ds_libero run \
  --config configs/first_episode.json \
  --calibration outputs/calibration-001 \
  --api-check outputs/api-check-001 \
  --out outputs/episode-001
```

每步打印进度。需要暂停时按 Ctrl+C；已写出的记录保留，回合记为未完成。首版不从回合中途恢复，因为完整复现还涉及控制器、夹爪内部状态和随机状态；重新运行使用新的输出目录和官方初始状态。代码、配置或环境变化后必须重做前置检查。

```bash
bash "$SIM" -B -m ds_libero report outputs/episode-001
```

报告和回放在运行结束时也会自动生成。硬中断后可用 `report` 恢复汇总；没有最终 `result.json` 的回合保持未完成。只需统计、无需 Pillow 时使用 `report ... --no-replay`。

## 固定协议

| 项目 | 首轮设置 |
|---|---|
| 仿真 Python / PyTorch | 3.10 / 2.5.1+cpu |
| MuJoCo / robosuite | 2.3.7 / 1.4.1 |
| LIBERO Git revision | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| 套件、任务顺序、任务、初始状态、seed | `libero_spatial`、0、0、0、0 |
| 指令 | pick up the black bowl between the plate and the ramekin and place it on the plate |
| 稳定步骤 | 10 步 `[0,0,0,0,0,0,-1]`，不计模型控制步数 |
| 控制 | 20 Hz 仿真控制间隔，最多 220 个模型动作 |
| 图像 | `agentview`、`robot0_eye_in_hand`，256×256，无损 PNG，原图垂直翻转一次 |
| 自身状态 | 世界坐标末端位置、xyzw 四元数、两根手指关节位置 |
| 历史 | 最近 4 步自身状态与已执行动作，无历史图像 |
| API | 官方 Chat Completions，`deepseek-flash`，thinking enabled，high，JSON mode |
| 上限 | 每响应 8192 tokens；HTTP timeout 120 秒；最多 2 次重试；每个命令最多 300 次 HTTP 尝试 |

`deepseek-flash` 是服务别名，不能保证未来仍对应同一权重版本。记录预期版本标签、请求/返回模型标识、日期、完整响应与 token 用量；返回别名本身也不能证明不可变版本。API 没有可验证的固定版本信息时，不宣称已 pin 住权重。

模型只输出：

```json
{"action": [0, 0, 0, 0, 0, 0, -1]}
```

前三维是世界坐标平移增量的归一化控制量，满幅目标偏移 0.05 米；接下来三维是世界坐标旋转向量增量，满幅 0.5 弧度；均在 `[-1,1]`。最后一维只能为 `-1`（开）、`1`（关）、`0`（保持夹爪控制量）。不额外归一化、不套用 VLA 动作统计、不自动裁剪、不转换为高层抓取脚本。位置控制器的目标偏移不等于一步内实际走过的距离。

模型仅得到任务指令、双视角图像、自身状态及最近动作。仿真物体位置、分割、目标判定和完整 MuJoCo 状态不进入模型请求。官方初始状态用于环境初始化，其哈希只用于审计。

等待 API 和重试期间不调用 `env.step()`。因此不要求 API 实时达到 20 Hz；实际运行速度与仿真时间分别报告。初步 128×128 冒烟速度约 4.73 步/秒，不是当前 256×256 配置的吞吐保证。

## 评分、异常与结果

- 每个有效动作执行一次，之后由 `env.check_success()` 判断完成；达到 220 个动作而未完成计失败。
- 空回复、截断、非法 JSON、重复键、额外字段、非有限数值、布尔值、越界值或非法夹爪值不会执行，记为 `invalid_output`（模型侧失败）。不会重新请求“修复动作”。
- 超时、429、5xx 对同一输入最多重试两次，退避 2/4 秒。鉴权与其他 HTTP 错误不重试。请求预算包含重试。服务可能已处理超时请求，日志中未知 usage 不视为零费用。
- 网络错误、预算中断、仿真异常、初始即成功的状态、异常环境终止和用户中断不形成模型成绩，`scored=false`、`task_success=null`；不会把 `done=true` 直接当成功。
- 进程退出码 0 表示得到了完整结果（也包括任务失败或模型输出失败），不代表任务成功；请看 `status` 和 `task_success`。基础设施错误与标定失败退出码为 2。
- 动作有效率的分母是获得的模型输出，不包含没有模型输出的网络失败；同时报告全部 HTTP 尝试、逻辑决策和 usage 覆盖率。reasoning tokens 属于 completion tokens 的子集，不能重复加总。

每个目录包含：

```text
manifest.json       配置、环境/代码/提示词哈希、Git revision、模型版本说明
prompt.txt          冻结的系统提示词
source_snapshot/    本次代码源文件快照
events.jsonl        观测、API 尝试、决策、动作前后状态及成功判定
responses/         去除密钥的原始 API 响应（含推理与 usage）
frames/            模型实际看到的双视角 PNG 与动作后观测
result.json        最终状态、是否计分、步数与耗时
summary.json       状态、动作有效率、token 与延迟分布等汇总
replay.gif         正式回合的双视角回放，按 20 Hz 仿真时间播放
```

`outputs/` 默认被 Git 忽略。小型汇总经检查后可复制到其他目录维护；不要提交密钥、原始响应或海量图像。首版不估算人民币费用，使用官方返回 token 与实际账单核对。

## 本地离线测试

核心逻辑测试使用标准库 unittest，不连接模型、不要求 LIBERO。图像相关测试需要 NumPy 和 Pillow；它们已在远程环境中安装。CI 使用与远程一致的版本。

```bash
python -m unittest discover -s tests -v
```

覆盖非法动作拒绝、网络重试不推进仿真、预算耗尽、官方成功判定、步数限制、原始图像单次翻转、前置检查不匹配、日志与 GIF 生成。该测试不能替代远程真实动作标定与真实 API 验收。

## 参考

- [LIBERO 环境](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01)
- [robosuite 1.4.1 OSC 配置](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/controllers/config/osc_pose.json)
- [OpenVLA LIBERO 评测步数](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py)
- [DeepSeek Vision](https://api-docs.deepseek.com/guides/vision/)
- [DeepSeek Thinking](https://api-docs.deepseek.com/guides/thinking_mode/)
- [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/)
