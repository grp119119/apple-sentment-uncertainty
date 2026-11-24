# v0.3 决策架构

## 数据流

```text
Text / Voice / Face / Physiology
                 |
                 v
      quality-weighted fusion
                 |
        +--------+--------+
        |                 |
        v                 v
  AffectEstimate   UncertaintyProfile
  distribution     entropy
  V/A(/D)          margin
                   quality/coverage
                   variance/temporal change
        |                 |
        +--------+--------+
                 v
         InterventionPolicy
        /                   \
EmotionOnly            UncertaintyAware
        \                   /
         v                 v
       Mode + Intensity + EpistemicStyle
                 |
                 v
       ESConv SupportAction
                 |
                 v
        Controlled Response
                 |
                 v
         Schema v3 Study Log
```

## 冻结边界

v0.3 正式实验界面的 `StudyFeedback` 仍只写入日志。v4.1 手动测试面板另行使用
`HistoryRecord` 保存使用者明确选择的 Need 和本轮反馈，并作为下一轮的状态输入。
History 只能在固定规则下排除被拒绝动作、保存偏好与重复次数；它不会在运行时修改
阈值、规则或 Policy 版本。

形成性研究完成后，需要：

1. 替换 `PilotThresholds`；
2. 更新 policy version；
3. 将 calibration status 改为 `FROZEN_V1`；
4. 保存冻结配置和测试快照；
5. 正式实验只读取冻结配置。

## 尚未宣称解决的问题

- 不宣称当前 fusion 权重最优；
- 不宣称 JSD 是 affective modality conflict 的最终指标；
- 不宣称阈值适用于真实设备；
- 不把自动识别结果当作临床诊断；
- 不使用反馈做在线强化学习；
- 不把 Context、History、Need 或 Persona 发展为并列主贡献。
- 不把手动输入的用户文字宣称为已经自动识别；本地 Need 模型尚未接入。
