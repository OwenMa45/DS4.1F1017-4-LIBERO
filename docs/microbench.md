# LIBERO-Five v1 微型 bench

五个既有任务现在有统一的 Python 接口和命令行入口。新实验实现 `create_policy(context)`，返回具有 `act(request)` 方法的策略，即可复用任务初始化、观测、动作检查、官方成功判定、日志和报告。原有 DeepSeek / GPT 配置也可通过同一入口运行。

实现位于 `ds_libero/microbench.py`，无需新 SDK；从仓库根目录导入即可。列任务、生成计划、读报告只需 Python 标准库。实际仿真仍使用已固定的远程 Python 3.10 / CPU / OSMesa 环境，不在 Windows 系统 Python 中启动 LIBERO。

## 固定任务与预算

| 序号 | 稳定任务键 | 内容 |
|---:|---|---|
| 0 | `libero_spatial:0` | 黑碗放盘 |
| 1 | `libero_object:0` | alphabet soup 汤罐入篮 |
| 2 | `libero_goal:0` | 开柜子中层抽屉 |
| 3 | `libero_goal:5` | 推盘到炉灶前方 |
| 4 | `libero_10:5` | 书本放入收纳格后格 |

沿用 LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`，任务顺序 0、每任务官方初始状态 0、seed 0、10 步稳定、20 Hz、双路 256×256 RGB 图像。运行时核对完整任务名。每任务一次试验；这是固定样本的微型诊断分数，不是完整 LIBERO 成绩或对其他初态的成功率估计。

默认使用最新对照的 **220 个实际控制步 / 最多 8 步动作块**，以及已验证的 grip_site 几何、两帧视觉历史、四步动作历史、抓取反馈和阶段切换边界。`MicroBench(max_steps=1024)` 使用之前的长时限预算；`action_horizon` 可设为 1..16，控制预算为 1..2048。

`protocol_sha256` 包含任务、初态、观测、动作执行规则和控制预算，不包含模型身份。因此相同控制配置的不同模型可比较；220 步与 1024 步的指纹不同。具体模型、提示词、checkpoint、推理设置另记在 `policy` 及完整配置中。还应核对各回合的运行环境记录，不能只凭一个指纹断言运行环境一致。

## 新实验：Python 接口

```python
from ds_libero.microbench import MicroBench
from examples.noop_policy import create_policy  # 换成自己的适配器

report = MicroBench(max_steps=220).evaluate(
    create_policy,
    out="outputs/my-experiment-001",
    policy_id="my-policy",
    revision="code-commit+prompt-v1+checkpoint-v1",
    policy_config={"checkpoint": "path-or-version"},
)
print(report["score"], report["coverage"])
```

在远程用 `bash "$SIM" -B your_experiment.py` 启动上述脚本。输出目录必须不存在。`examples/noop_policy.py` 是可执行的接口示例：保持机械臂并张开夹爪，不调用模型，也不宣称能完成任务。

最小适配器形状：

```python
from ds_libero.microbench import EpisodeContext, PolicyInput

class MyPolicy:
    def __init__(self, context: EpisodeContext):
        self.context = context
        # 在这里创建自己的模型会话。

    def act(self, request: PolicyInput) -> list[list[float]]:
        # 用模型预测替换这条保持动作。
        return [[0., 0., 0., 0., 0., 0., -1.]]

    def close(self):  # 可选；成功、异常和中断后都会调用
        pass

def create_policy(context: EpisodeContext):
    return MyPolicy(context)
```

每任务调用一次 factory；策略自行管理的跨任务全局变量需要自行重置。可以共享只读模型权重，不能延续上个任务的动作缓存或会话历史。

`EpisodeContext` 提供任务标识、自然语言指令、控制预算/频率、`control_instructions`（参考控制约定）和一份独立的 `policy_config`。bench 不会替自定义模型决定提示词。自定义适配器需在 revision / policy_config 中记录实际提示词和预处理版本。配置会落盘，不应包含凭据；密钥由适配器从环境变量读取。

`PolicyInput` 的字段：

| 字段 | 含义 |
|---|---|
| `instruction` | 来自实际任务的自然语言指令 |
| `observation.images` | `agentview` / `robot0_eye_in_hand` 的已转正 PNG **bytes**；自行解码，不再上下翻转 |
| `observation.state` | 末端世界坐标、xyzw 四元数、夹爪关节位置及几何校正后的自身状态 |
| `observation.geometry` | 相机标定、机器人自身几何；没有物体真值、分割或成功标记 |
| `history` | 最近实际执行的动作与前后自身状态，最多 4 步 |
| `visual_history` | 最近最多两次决策的 `(step, Observation)`，首步为空 |
| `control_feedback` | 上次实际执行的前缀、丢弃动作数及测得的机器人运动 |
| `step` / `remaining_steps` | 已执行 / 剩余控制步数 |
| `max_actions` | 此次允许返回的最大动作数，已考虑剩余步数 |

输入是独立拷贝；适配器不接收环境对象或仿真内部状态。返回 Python `list[list[float]]`，长度 1..`request.max_actions`，即使仅一步也需要外层列表。NumPy / Tensor 输出须由适配器显式转换为 `.tolist()`。

每条动作严格为 7 维：前三维为世界坐标平移增量（每维归一化至 `[-1,1]`，缩放 0.05 m），接下来三维为世界坐标旋转向量增量（缩放 0.5 rad），最后为 `-1` 开 / `0` 保持夹爪目标 / `1` 关。不自动裁剪、补齐或将动作块截短以修复格式。整个动作块先验证，超过剩余预算也会拒绝；阶段边界只执行合法前缀，剩余后缀丢弃并重新观测。

返回非法动作或抛出 `InvalidAction` 记为模型输出失败；API / 服务异常可抛 `ds_libero.errors.APIError`，请求预算耗尽可抛 `BudgetExhausted`，均保留为未评分。其他异常记录为 `runtime_error`。自定义模型的网络超时、重试和调用预算由适配器负责；bench 只统一控制步数，不会自动重试自定义推理。等待推理不会推进仿真。

## 命令行入口

以下离线命令可在本地执行，不需要 API 密钥，也不会导入适配器：

```bash
python -m ds_libero microbench list
python -m ds_libero microbench plan \
  --policy examples.noop_policy:create_policy \
  --policy-id noop-interface-check --revision v1 \
  --max-steps 220 --out outputs/microbench-noop-001
```

`--policy` 使用可导入的 `module:callable`。`--policy-config options.json` 可传非敏感配置。也可用 `--config configs/microbench_220.json`，此时不能同时设置 `--max-steps`。

在远程仓库根目录生成计划并运行，或把本地计划随相同版本代码带到远程：

```bash
SIM=/mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh
bash "$SIM" -u -B -m ds_libero microbench run outputs/microbench-noop-001
```

对自己的实验，替换 `--policy` 和身份字段后重新 plan。实际运行先为每项做动作标定，再重置官方初态并评测。外部适配器的第一个输出在正式回合中校验，不额外发送一次不执行的推理请求。

已有 API 实验通过配置直接接入，沿用其密钥来源、请求格式、独立 API-check、重试和费用统计：

```bash
python -m ds_libero microbench plan \
  --config configs/gpt61_sol_medium_220.json \
  --out outputs/microbench-gpt-001
bash "$SIM" -u -B -m ds_libero microbench run outputs/microbench-gpt-001
```

DeepSeek 长时限实验将配置换为 `configs/direct_v2_geometry_1024_high.json`。运行 API 配置会产生模型调用；plan、list 和 report 不会。

## 报告和恢复

```bash
python -m ds_libero microbench report outputs/my-experiment-001
# 只执行尚未开始的第 3 项；总体报告仍保留全部五项作为分母
bash "$SIM" -u -B -m ds_libero microbench run outputs/my-experiment-001 --episode 2
```

Python 中对应 `MicroBench.report(directory)` 和 `MicroBench.run(directory, policy_factory=create_policy)`。可导入的 factory 已记入计划时，`run` 可省略 factory 参数。`run(..., episode=2)` 选择单项，`replay=False` / `--no-replay` 不生成 GIF，但继续保存原始图像与日志。

- `plan.json` / `configs/`：冻结任务、完整配置、协议指纹和策略身份。
- `policy_source.json`：首次运行时记录外部 factory 的模块名与所在文件哈希，后续执行检查一致性；这不包含依赖模块或模型权重，其版本仍由 revision 声明。
- `microbench.json`：稳定公共报告，包含 `score`、`coverage`、逐任务成功/状态、控制步数、决策次数、有效率、延迟和已知成本。
- `scores.csv`：每任务一行；`aggregate.json` / `episodes.csv` 保留原有详细报告。
- `runs/episode-0000/` 至 `0004/`：标定、可选 API-check、正式回合、逐步记录、响应、PNG 和 GIF。

五项全部可评分时，`score = successes / 5`，范围 0..1。官方成功为 1；用完控制步数未成功、正式回合 `invalid_output` 为 0。基础设施错误、初始即成功、异常 `done`、中断、请求预算耗尽、API-check 失败和未开始项均不伪装成失败：有任何未评分项时 `score=null`，同时报告覆盖率与五项样本的成功率上下界。`batch_exit_code=0` 表示选定任务已处理，不表示任务成功或五项都可评分；CLI 错误/中断退出 2。

外部适配器的推理调用数、token 与费用在公共报告中为 `null`，`inference_accounting=not_instrumented`；其端到端决策延迟照常记录。旧格式的原始报告只统计 bench 实际记录到的 API 事件，不能据其中的零推断外部服务免费。内置 API 的正式回合成本与包含 API-check 的 campaign 成本分开提供。

重复 run 跳过已经记录的完整结果，不自动重新付费。错误或中断的已有目录会阻止自动继续；保留记录，显式选另一未开始项，或创建新实验目录重跑。每项结束后刷新报告。代码、配置或策略发生变化应创建新目录；不支持从中断回合的中间状态恢复。

## 验证

```bash
python -m unittest discover -s tests -p test_microbench.py -v
python -m unittest discover -s tests -v
```

microbench 接口测试只用标准库和显式仿真替身，不调用模型。覆盖五项固定选择、模型隔离、观测隔离、预算、动作块验证、阶段边界、官方成功即停、异常和覆盖率、前置检查、恢复与配置漂移。完整回归还需原有 NumPy / Pillow 依赖。离线测试通过不代表真实模拟器标定或任务成功。
