# DeepSeek + SAM3：LIBERO-Five 第一个工具实验

本实验把本地 SAM3 注册为 DeepSeek 的 `sam3_segment(camera, prompt)` 工具。DeepSeek 从当前双相机图像和任务指令选择目标短语、相机，harness 调用 SAM3，再把预测框、掩膜质心、置信度和带编号的掩膜叠加图作为工具结果交给 DeepSeek，最终仍输出原有 7D `actions`。

这次交付包含代码、离线验证和服务器运行入口；**尚未在你的 GPU 上加载权重、调用真实 DeepSeek 或测得任务成功率提升**。离线测试使用明确的模型/环境替身。

## 运行环境与路径

- SAM3 包目录：`/mnt/cpfs/users/mrq/package/sam3/sam3`。导入时加入 Python 搜索路径的是它的父目录 `/mnt/cpfs/users/mrq/package/sam3`。
- 权重目录：`/mnt/cpfs/users/mrq/package/sam3/checkpoints`。优先选 `sam3.pt`；否则只在恰有一个 `.pt` / `.pth` 文件时自动选择。多个候选必须传 `--checkpoint /绝对路径/文件.pt`。
- 默认词表：包目录中的 `assets/bpe_simple_vocab_16e6.txt.gz`；位置不同时用 `--bpe FILE`。
- 权重按官方 SAM3 的 `model` / `detector.*` 格式提取 image detector，并做严格键/形状校验；也接受完整 image model state dict。文件与本地代码不匹配时停止，不使用部分随机初始化的模型继续跑分。
- SAM3 使用你已准备的 GPU Python，常驻一个独立进程。默认 CUDA + bfloat16；不支持 bfloat16 的 GPU 可以传 `--precision float32`。用 `CUDA_VISIBLE_DEVICES` 选择显卡。
- LIBERO 继续由 `/mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh` 启动。**不要向它的固定 Python 3.10 / CPU 环境安装 SAM3 或替换 torch。** 仿真端仅使用已有 Pillow 和标准库 HTTP 客户端。
- 服务仅监听 `127.0.0.1:8765`，默认要求服务与仿真在同一节点；跨节点时用 SSH 本地端口转发。不会自动下载权重，找不到本地文件会直接失败。

SAM3 接口按[官方 image builder](https://github.com/facebookresearch/sam3/blob/main/sam3/model_builder.py)和[image processor](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py)核对。你的本地版本若不同，启动失败时请回传 worker 日志，不要修改仿真环境来迁就它。

## 没有 Conda：复用节点加速器栈创建 venv

对用户当前的 Python 3.12.3 / PPU-ZW805 节点，`nvidia-smi` 显示的是 PPU-SMI / HGGC 信息，不应据此安装或升级公共 NVIDIA CUDA torch wheel。提供了 `scripts/setup_sam3_env.py`：用 `venv --system-site-packages` 继承节点现有包，在新环境内补 SAM3 依赖，不改动 LIBERO 或基础 Python。该环境依赖当前节点的基础解释器和加速器库，不能搬到另一种节点直接使用。

```bash
cd /mnt/cpfs/users/mrq/DS4.1F1017-4-LIBERO
git pull --ff-only origin codex/direct-policy-v2
/usr/local/bin/python scripts/setup_sam3_env.py \
  --env /mnt/cpfs/users/mrq/envs/sam3 \
  --sam3-repo /mnt/cpfs/users/mrq/package/sam3 \
  --out outputs/sam3-env-001
```

脚本先验证基础 torch / torchvision 导入及 `torch.cuda` 接口，再创建环境。根据**你本地** SAM3 的 `pyproject.toml` 解析依赖，固定已装加速器包版本，先审查 pip dry-run：任何将安装 torch、torchvision、Triton、PPU、NVIDIA/CUDA 包的计划都会停止。实际依赖安装使用审核后的版本和 `--no-deps`；SAM3 本体也使用 `--no-deps --no-build-isolation`。安装后再次校验加速器模块路径和版本，最后仅做 SAM3 导入及 2×2 矩阵运算，不加载模型权重、不调用 DeepSeek。

看到 `READY` 后，不必激活环境，显式指定解释器即可：

```bash
export SAM3_PYTHON=/mnt/cpfs/users/mrq/envs/sam3/bin/python
set -o pipefail
bash scripts/start_sam3.sh --precision float32 2>&1 | tee outputs/sam3-worker-002.log
```

先以 `float32` 验证 PPU 上的真实前向推理，再另建实验评估 bfloat16；成功创建环境不代表所有 SAM3 算子均已在 PPU 上验证。加速器仍通过框架兼容的 `cuda` 接口使用，不安装 NVIDIA 驱动。也可 `source /mnt/cpfs/users/mrq/envs/sam3/bin/activate` 进入环境。

脚本不会覆盖已有环境或日志目录；失败时先回传 `/mnt/cpfs/users/mrq/DS4.1F1017-4-LIBERO/outputs/sam3-env-001/environment.json` 和其中失败阶段对应的 `.log`，不要在基础 Python 上逐个重装 torch。若基础环境探测失败，需要节点的厂商运行时支持，不能仅靠 venv 修复。本脚本的保护逻辑经过离线测试，PPU 实机结果以你的返回为准。

## 1. 更新仓库并启动 SAM3（终端 A）

先激活你准备好的 **SAM3 环境**，然后执行：

```bash
cd /mnt/cpfs/users/mrq/DS4.1F1017-4-LIBERO
git fetch origin
git switch codex/direct-policy-v2
git pull --ff-only origin codex/direct-policy-v2

# 此处 python 必须是你的 SAM3 环境解释器。
export SAM3_PYTHON="$(command -v python)"
"$SAM3_PYTHON" -c 'import sys, torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available())'
mkdir -p outputs
set -o pipefail
bash scripts/start_sam3.sh 2>&1 | tee outputs/sam3-worker-001.log
```

等待出现 `SAM3 ready at http://127.0.0.1:8765`，保留终端 A。加载时计算权重、SAM3 源码和词表 SHA256，首次启动需要时间。也可显式设置 `SAM3_PYTHON=/你的SAM3环境/bin/python`。上述脚本的附加参数会传给服务，例如：

```bash
bash scripts/start_sam3.sh --checkpoint /mnt/cpfs/users/mrq/package/sam3/checkpoints/sam3.pt --precision float32
```

## 2. 先验证一项任务（终端 B）

```bash
cd /mnt/cpfs/users/mrq/DS4.1F1017-4-LIBERO
SIM=/mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh
OUT=outputs/deepseek-sam3-220-001
set -o pipefail

# 无 DeepSeek 费用：真实 SAM3 前向推理 + HTTP + PNG + 结果落盘检查。
bash "$SIM" -u -B -m ds_libero.sam3_server check \
  --config configs/deepseek_sam3_220.json --out "${OUT}-sam3-check"

# 离线冻结五任务，不调用模型。
bash "$SIM" -B -m ds_libero microbench plan \
  --config configs/deepseek_sam3_220.json \
  --policy-id deepseek-v4.1-flash-1017-sam3 --revision sam3-tools-v1 --out "$OUT"

# 使用既有 DEEPSEEK_API_KEY 或项目根目录 ds_API.txt。
# 这一项会依次执行动作标定、真实 API+SAM3 检查、黑碗正式回合。
bash "$SIM" -u -B -m ds_libero microbench run "$OUT" --episode 0 \
  2>&1 | tee "${OUT}-episode0.console.log"
```

`sam3-check` 默认用空白图，仅验证模型确实能前向运行和完整传输；没有检测到物体是正常结果，不表示视觉准确。可额外加 `--image /已有回合/frames/00000_agentview.png --prompt 'black bowl'` 检查实际图像，必须是已经正向的 256×256 单相机帧。正式 `api-check` 自动使用当前任务的真实相机图像，返回动作不执行。

先检查 `"${OUT}/runs/episode-0000/api-check/result.json"` 的 `status=pass`，以及 `api-check/sam3/step_00000_call_0001/overlay.png` 是否对齐。`microbench run` 会自动把前置失败留在日志中，不跳过门禁。

## 3. 跑完剩余四项，或一次运行全部

已按上节完成黑碗后，继续同一个冻结计划；已记录任务不会重复收费重跑：

```bash
bash "$SIM" -u -B -m ds_libero microbench run "$OUT" \
  2>&1 | tee "${OUT}-remaining.console.log"
bash "$SIM" -B -m ds_libero microbench report "$OUT"
```

已有中断/基础设施错误的任务不自动重跑；先保留并分析结果。可用 `--episode 1` 等显式选择后续未开始任务。修改代码、阈值或权重后必须建新输出目录、新计划，不复用旧计划。

若希望从零一次跑完五项，在 worker 就绪后用新的输出名：

```bash
bash scripts/run_sam3_five.sh outputs/deepseek-sam3-220-all-001
```

## 4. 对照与预算

| 设置 | 基线 | SAM3 实验 |
|---|---|---|
| 配置 | `configs/deepseek_baseline_220.json` | `configs/deepseek_sam3_220.json` |
| 五任务、初态、seed | 同一固定选择，init=0、seed=0 | 相同 |
| 每任务控制步 / 动作块 | 220 / 最多 8 | 相同 |
| 几何、抓取反馈、短视觉历史 | 开启 | 相同 |
| DeepSeek | `deepseek-flash`、high、max_tokens=null | 相同 |
| 感知 | 原始双视角 RGB | RGB + 模型主动调用 SAM3 |
| 每个控制决策工具预算 | 无 | 最多 2 轮，每轮最多 4 次分割 |
| SAM3 检测 | 无 | 阈值 0.35，每调用最多 6 个实例 |

默认 `require_call=true`：每次控制决策至少调用一次 SAM3，否则输出按本实验契约视为 `invalid_output`。工具选择使用 `auto`，通过系统提示词要求调用；[DeepSeek thinking 模式不支持 `required` 或指定工具名称的强制选择](https://api-docs.deepseek.com/api/create-chat-completion/)。如需研究“可选使用工具”，改为 `false` 并新建计划，它是另一项策略消融。

最多 2 轮工具调用后，使用 `tool_choice=none` 和 JSON mode 要求最终动作；任一 HTTP 重试仍计入同一请求预算。默认每回合最多 1980 次 HTTP 尝试，覆盖极端情况下 `220 × 3 轮请求 × 3 次尝试`，并非预先发送这些请求。API-check 单独最多 9 次，所以单任务含 preflight 的上限为 1989。基线设相同回合请求上限，实际通常远少于此。`max_tokens=null` 沿用已有 high 基线，不限制服务端默认输出；费用以日志为准。

单个动作块未执行完时不会再调用工具。等待 SAM3、DeepSeek 或重试均不推进仿真。每次 `act` 从新观测构造对话，仅在该次决策内部保留完整工具消息与 [`reasoning_content`](https://api-docs.deepseek.com/guides/thinking_mode/)。

运行基线：

```bash
BASE=outputs/deepseek-baseline-220-001
bash "$SIM" -B -m ds_libero microbench plan \
  --config configs/deepseek_baseline_220.json --policy-id deepseek-baseline --out "$BASE"
bash "$SIM" -u -B -m ds_libero microbench run "$BASE"
```

两组的 `protocol_sha256` 应相同，策略配置/提示词不同。对照成功率时同时看决策次数、DeepSeek token/费用、SAM3 调用次数和总耗时；它们没有被约束为相同推理算力。五次固定初态只构成微型诊断，不等于完整 LIBERO 成绩。模型别名和预期版本标签也不能证明远端权重不可变。

## 观测边界与审计

- 工具只可选择两路当前相机和一个文本短语；不接受任意图像路径、URL、历史帧或仿真状态。RGB 只在仿真适配器中翻转一次，SAM3 和叠加图不再翻转。
- 结果只含预测的二维坐标，不提供物体真值、深度或 3D 抓取点。实例 ID 只在本次调用内有效；掩膜中心不保证是抓取点，不能当作跨视角对应点自动三角化。
- SAM3 找不到目标时返回空列表；服务故障、超时、格式不兼容或运行时变化中止任务并保留为未评分，绝不伪装为空检测或静默退回基线。
- `sam3_runtime.json` 固定本次五任务的权重、代码、词表、依赖和精度信息；每阶段以及每个结果都会核对 runtime ID。
- 每个 API 子轮有独立且可校验的 `requests/*.json` 和原始 `responses/*.txt`。所有子轮/重试都进入既有 API 用量与费用统计。控制决策和工具子轮不混为一个计数。
- 每阶段的 `sam3/step_*_call_*/` 保存 `result.json`、`overlay.png`、`mask_*.png`。`events.jsonl` 包含工具参数、当前帧 SHA256、耗时和异常；`summary.json` / `microbench.json` 包含 `sam3` 统计。GPU 费用未知，记为 `null`。
- 现有 `replay-request` 只重放一个 HTTP 请求，不会执行 SAM3 工具链；需要重现完整决策时应检查该决策的各轮请求与工具产物，或在新计划重跑。旧基线的请求/配置指纹保留。

## 回传文件

优先回传：`microbench.json`、`scores.csv`、`sam3_runtime.json`，每任务 `api-check` / `episode` 下的 `result.json`、`summary.json`、`events.jsonl`，以及 SAM3 叠加图。遇到报错，加上对应控制台日志和 `responses/`。以下归档保留这些诊断材料，排除体积较大的原始帧、请求图像数据及每实例二值掩膜；服务器原文件不会删除：

```bash
# 沿用上文 OUT，文件不存在时可省略相应日志参数。
tar --exclude='*/frames' --exclude='*/requests' \
    --exclude='*/sam3/*/mask_*.png' \
    -czf "${OUT}-results.tar.gz" \
    "$OUT" "${OUT}-sam3-check" outputs/sam3-worker-001.log \
    "${OUT}-episode0.console.log"
```

若启动 worker 就失败，先回传 `outputs/sam3-worker-001.log`。若 smoke 失败，回传 `${OUT}-sam3-check/result.json` 和该目录日志即可。不要把 `ds_API.txt` 或环境变量转储加入归档。

## 离线验证

```bash
python -m unittest discover -s tests -v
python -m ds_libero.sam3_server --help
```

核心测试覆盖原有控制约定兼容性、五任务计划、thinking 工具上下文、HTTP/工具预算、非法输出不执行、服务故障不计模型失败、请求日志不覆盖、掩膜像素坐标/方向、服务通信与结果落盘。需要 NumPy 和 Pillow，无需 torch、SAM3 权重或 API 密钥；这些测试不能替代上述真实 GPU/API 验收。
