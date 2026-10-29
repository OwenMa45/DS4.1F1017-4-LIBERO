# Panda EEF、指垫与相机几何核对

2026-10-29。新增 `configs/direct_v2_geometry_high.json`，以 Astranomy §3.1.2 的接口说明思路完善 DS 提示词，同时修正本项目观测参考系不一致的问题。新实验仍输出原生 7D 增量动作；high、不指定客户端 token 上限、逐任务开销及 latency 统计保留。

## 发现与来源

核对固定 LIBERO revision `8f1084e3132a39270c3a13ebe37270a43ece2a01` 及 robosuite 1.4.1：

1. `SingleArm._setup_observables` 的 `eef_pos` 读 `grip_site` 世界位置，`eef_quat` 却读 `robot_model.eef_name`，即 `robot0_right_hand` body 的世界姿态。控制器加载时 `eef_name` 指向 `grip_site`，控制器的姿态来自该 site 的 `site_xmat`。[SingleArm](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/robots/single_arm.py#L123)、[观测](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/robots/single_arm.py#L304)、[控制器状态](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/controllers/base_controller.py#L141)。
2. Panda gripper 的根 body 相对 hand 绕 Z 轴旋转 -90°；`grip_site` 沿 hand 的 +Z 偏移 0.097 m。因此旧观测的位置与四元数不是同一坐标架。这个偏差会影响从四元数解释局部几何；它不意味着所有世界轴增量动作都旋转错了。旧的世界增量方向标定也不一定能发现该差异。[夹爪 XML](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/models/assets/grippers/panda_gripper.xml)。
3. XML 中的指垫 box 中心位于 hand 局部 z=0.0934 m，故在控制 site 局部为 z=-0.0036 m。左右中心分别为 `[-q1-0.0035, 0, -0.0036]` 和 `[-q2+0.0035, 0, -0.0036]` 米，q1/q2 是两侧滑动关节位置。这些是机器人几何中心，不是目标位置或接触点。夹爪开合轴是 site 局部 X，向指尖延伸是局部 +Z。
4. Panda 原始腕相机相对 hand 偏移 `[0.05,0,0]` m，fovy=75°；从该 XML 推导，其相对控制 site 的位置为 `[0,0.05,-0.097]` m，光学视线沿 site +Z。这与 DROID 的腕相机倾角不同。实际运行使用加载后相机的世界外参，不硬编码这些相机数值。[机器人 XML](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/models/assets/robots/panda/robot.xml#L236)。
5. robosuite 默认 RGB 使用 OpenGL 行顺序，适配器翻转一次得到正向 PNG。MuJoCo 相机旋转右乘 `diag(1,-1,-1)` 后，光学坐标为 X 右、Y 下、Z 前；K 使用单幅 256×256 图像像素。新配置拒绝不匹配的原始图像约定。[图像 sensor](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/environments/robot_env.py#L389)、[macros](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/macros.py#L28)。

LIBERO 通过 [env_wrapper.py](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/env_wrapper.py) 使用 Panda/OSC_POSE，本文几何以运行时加载的机器人具名 site/body/geom 核验，不读取物体状态。

## 新配置行为

robosuite 的观测按采样频率缓存，而直接读取的 MuJoCo 几何属于当前内部状态。新配置在每次对外观测时先 `sim.forward()`，再强制刷新全部观测，并检查模拟时钟未变；然后才读取图像、本体状态和几何，避免新旧时刻混用。它不额外执行控制动作，但会增加观测开销，该开销包含在现有实际任务耗时中。[固定版本观测流程](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/environments/base.py#L326)。

`geometry_contract=true` 仅用于 v2，要求开启相机和抓取反馈。每次观测核对 controller、observable 的 site 一致，检查原始位置、hand 姿态、site 安装变换和指垫中心。误差阈值为位置 1e-5 m、旋转矩阵元素 1e-5；这用于检测模型/坐标架不匹配，不是抓取精度指标。

然后将 `eef_quaternion_xyzw` 修正为 `grip_site` 世界姿态，使其与 `eef_position_world_m` 一致。旧 hand 四元数保存在 `camera_geometry.control_frame.legacy_hand_quaternion_xyzw` 供审计。旧配置保持原观测、提示词和配置指纹，作为历史基线；新实验使用新配置。

新增提供给 DS 的信息：

- `control_frame.R_world_from_control` 及坐标架名称。
- 指垫在控制坐标架内的位置、左右指垫连线方向、指尖延伸方向。
- 各相机在控制坐标架内的位置、光学旋转和世界视线方向；历史帧使用各自同时刻的数据。

提示词分清控制参考点、指垫几何中心、关节间距和图像中心，给出变换与投影的用法，强调单像素不等于三维定位。控制目标仍每步相对实际位置更新；非零旋转左乘实际 site 姿态，零旋转保持之前的姿态目标。这一细节来自 [OSC set_goal](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/controllers/osc.py#L257)。

当前抓取反馈和动作阶段边界保持不变。由于本次同时修正观测语义、增加几何字段并修改提示词，它属于接口修正版，不能把与旧回合的差别全部归因于提示词文字。

## 标定及验证

新增离线测试使用固定 XML 的独立前向运动学，覆盖不同 hand 姿态、非对称夹爪开口、接近 180° 的旋转、腕相机投影、原始图像单次翻转、同步刷新且不推进仿真时间、参考系不匹配拒绝、旧配置兼容，以及新配置的任务/流水线费用和 latency 统计。

远程标定仍先于任何付费 API。每张标定快照保存几何误差，并在 `calibration/geometry_overlays/` 生成诊断图：E=控制 site，L/R=指垫中心，M=中点。它们独立于 `frames/`，不会替换模型输入。观察标记应落在对应机器人几何位置；指垫 geom 中心未必是可见表面或接触面，E 与 M 接近是正常的。

离线测试验证软件与固定模型几何一致性；真实远程渲染、接触和抓取效果仍由下一轮标定与回合验证。

## 下一轮远程命令

```bash
cd /mnt/cpfs/users/mrq/DS4.1F1017-4-LIBERO
git switch codex/direct-policy-v2
git pull --ff-only origin codex/direct-policy-v2

bash scripts/run_pilot.sh \
  configs/direct_v2_geometry_high.json \
  outputs/pilot-v2-geometry-high-001
```

原 `ds_API.txt` 或环境变量方式继续有效；不改动远程 Python/仿真依赖。新配置与代码需要重新标定，不复用旧 gate。若标定不通过，脚本停止，不发起付费 API；提供 `calibration/summary.json` 和控制台日志核查。

成功完成流水线后，提供 `campaign_summary.json`、`episode/summary.json`、`episode/events.jsonl`、`episode/replay.gif`，另附 `calibration/summary.json` 及几张初始/旋转后的双视角投影图。

`episode/summary.json` 继续记录每个任务的 API usage、估算人民币区间、HTTP/decision 延迟和实际任务墙钟；`api_calls.json/csv` 是逐请求明细；`campaign_summary.json` 包含前置检查费用。失败与不完整请求也保留已知开销，未知值不作零。HTTP、decision 与任务墙钟相互重叠，不相加。多任务运行仍可通过冻结计划按 task 汇总。

设计参考：[Astranomy §3.1.2](https://research.nvidia.com/labs/srl/projects/astranomy/#results-prompt)。本项目本地知识库不在此提交中。
