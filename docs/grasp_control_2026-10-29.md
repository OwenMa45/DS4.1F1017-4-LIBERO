# 首个完整回合的抓取诊断与改进

核对用户提供的 summary、campaign_summary、220 步 events 和 221 帧 GIF 后，确认流程完整运行，但任务结果是 `failure / control_step_limit`，不是成功抓取。37 次 API 均为 stop、合法动作，无截断或重试。

正式回合消耗 932,870 tokens（729,962 reasoning，属于 completion 的子集），实际任务耗时 4039.892 s（67.33 分钟），API 等待合计 3977.399 s，仿真时间 11 s；人民币估算区间 3.121600–6.243200 元。含前置检查共 38 请求、951,148 tokens、3.189168–6.378336 元；三个命令耗时合计 4275.140 s，不含命令间隙和 GIF 生成。

## 证据及边界

日志的 step 为从 0 开始的动作索引，GIF 的 step 为已经执行的动作数。

| 动作片段 | 已观测事实 |
|---|---|
| step 32–33 闭合，34 开始抬升 | 只静止闭合 2 步即 0.1 s，抬升前两指关节间距仍为 74.55 mm |
| step 56 闭合，57 开始抬升 | 仅静止闭合 1 步；间距反而从 38.73 增至 45.45 mm，指令切换不等于机械到位 |
| step 108–110 闭合，111 抬升 | 静止闭合 3 步后间距仍为 53.52 mm |
| step 196 闭合，197 抬升 | 静止闭合 1 步后间距为 75.07 mm |
| 后续多个搬运片段 | 间距逐渐降至约 2 mm，但回放未显示稳定带起目标碗 |

重复盲目执行接近、短暂闭合、抬升、横移，是可直接从日志确认的问题。单凭关节间距无法判定抓住了哪件物体或是否碰撞；早抬升也不能证明是唯一原因。具体抓取深度与平面对齐仍需视觉验证。当前有相机标定和 EEF 投影，但腕部图像中心不能直接当作抓取点。

[robosuite 1.4.1 PandaGripper](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/models/grippers/panda_gripper.py) 对目标逐渐积分：+1/-1 推进闭合/张开，0 保留当前目标，不继续推进关闭。机械运动仍有响应延迟，不能把一条 close 当作抓取已完成。[官方夹爪模型](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/models/assets/grippers/panda_gripper.xml)定义的指垫几何中心可作为机器人自身位置参考，不代表物体接触点。

## 实现

新配置 `direct_v2_grasp_high.json` 增加两项开关：

- `grasp_feedback=true`：向 DS 提供当前关节间距、上次实际执行前缀及模拟时长、真实末端位移、间距变化；同时提供两侧机器人指垫中心、其中点及相机投影。只读取具名的机器人指垫几何，不读取物体位置、接触标签、分割或成功状态。
- `gripper_feedback_barrier=true`：在同一批动作第一次出现夹爪开/合方向反转，或机械臂移动与纯夹爪动作相互切换时，只执行切换之前的前缀。剩余动作丢弃，模型看新观测再决定。至少执行一条；整批动作仍先严格校验，非法尾部不能通过“只执行前缀”绕过校验。

例如 DS 输出“闭合、闭合、抬升、抬升”：先执行两条闭合，再交给 DS 新图像和间距。DS 可以继续闭合、修正位置或选择抬升；执行器不会替它闭合固定时长、生成恢复动作或判定抓取成功。若 DS 在一条动作内同时移动和闭合，这条动作仍按原样执行；边界机制不保证它一定闭合足够久。

提示词强调按观测进行对齐、充分闭合、小幅测试抬升、观察物体是否随动后再搬运。这是通用反馈指导，没有该任务的目标坐标或固定抓取路径。新实验仍为 direct policy，但明确记录了与原 v2 不同的观测和执行协议，不当作只修改提示词的严格单因素比较。

保持 high、不指定客户端 token 上限、相同初始状态/seed、20 Hz、220 步、每次最多 8 条原生 7D 动作。原配置和指纹不变；`v2_grasp_feedback_only_high.json` 可在需要时单独测试反馈/提示词、不启用执行边界，避免同时修改的效果无法区分。

标定新增每一步开合的关节间距轨迹，便于确认真实响应速度。正式报告新增 `gripper_motion_diagnostics`、`feedback_boundaries`、`discarded_actions_at_feedback_boundaries` 和 `executed_actions_per_decision`，区分模型提议的动作和实际执行动作。现有 API 开销、前置检查费用及实际 latency 统计继续保留。更多反馈边界可能增加 API 次数和耗时；并不承诺更便宜或更快。

## 下一轮远程命令

从同一官方初始状态重新运行，不接着失败回合的末端状态恢复。根目录 `ds_API.txt` 已可读取；不要上传该文件。

```bash
cd /mnt/cpfs/users/mrq/DS4.1F1017-4-LIBERO
git switch codex/direct-policy-v2
git pull --ff-only origin codex/direct-policy-v2

bash scripts/run_pilot.sh \
  configs/direct_v2_grasp_high.json \
  outputs/pilot-v2-grasp-high-001
```

脚本重新标定、检查 API，再运行一个回合；旧 gate 与新代码/配置不匹配，不复用。对照时保留旧失败回合，不能替换原成绩。

结束后提供 `episode/summary.json`、`episode/events.jsonl`、`episode/replay.gif`、`campaign_summary.json`。若标定未通过，提供 `calibration/summary.json`，其中现在有完整开合响应轨迹。

下一次优先检查：是否真正带起目标碗、闭合后是否利用反馈修正、是否减少空夹后搬运；再比较成功率、API 用量和耗时。单个回合的改善不等于 LIBERO 整体性能提升。
