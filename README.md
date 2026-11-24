# 情感感知不确定性介入原型 v4.1

v4.1 在中文数值模拟测试面板中新增手动 `UserNeed` 和确定性的 `HistoryRecord`
反馈闭环。使用者可以输入七类情绪概率、效价、唤醒度、疲惫程度、信号质量、
数据覆盖率和交互情境，并手动选择倾听、安慰、梳理确认、建议、信息、暂停或
未明确需求。用户原始文字当前只记录，不进行自动识别；所有输入仍属于演示数据。

每轮运行后可以手动记录是否接受以及改善、不变、变差或暂不评价。History 会在
下一轮排除被拒绝的支持动作、保存明确需求和重复次数，但不会修改 Policy 阈值或
规则。疲惫程度当前只记录，不参与决策。

v0.3 核心算法、阈值和测试保持不变，`main_v3.py` 仍可独立运行。

v0.3 是项目根目录的完整新架构。它实现：

```text
Multimodal Observation
        -> Affect Estimate + Uncertainty Profile
        -> Intervention Mode + Intensity
        -> ESConv Support Action
        -> Response + Experiment Log
```

## 科学依据与边界

- 逐轮情绪采用 Ekman 六类加 Neutral，并保存完整概率分布。
- ESConv 七类负性情绪只用于场景/参与者自报，不作为自动诊断。
- uncertainty 包含 predictive entropy、top-two margin、signal quality、coverage、
  ensemble variance 和 temporal instability。
- ECE 与 Brier score 用于离线概率校准。
- support action 保持 ESConv 原始八类。
- policy 以 JITAI 的 decision point、tailoring variable、intervention option、
  decision rule 和 outcome 结构组织。

`cross_modal_disagreement` 目前仍是实验性指标。系统会计算它，但明确标记为需要
affect-specific calibration。`PilotThresholds` 中的所有数值都是工程默认值，状态为：

```text
PILOT_DRAFT_NOT_VALIDATED
```

它们不能作为论文正式阈值。Study 1 后应替换并冻结为 Policy v1.0。

## 两个实验条件

### EmotionOnlyPolicy

- Affect 决定 support action 和基础强度；
- 忽略 perceptual uncertainty；
- 除共同安全/可接受性边界外，始终直接介入。

### UncertaintyAwarePolicy

- 证据不足且用户未主动发起：`WAIT`；
- 证据不足/高度模糊且用户主动发起：`CONFIRM`；
- 中等不确定：`TENTATIVE`；
- 低不确定：`DIRECT`；
- 用户不接受或安全边界阻止：`DEFER`。

两者共用相同的 Affect fusion、support action selector 和 response resources，保证正式实验的
核心差异只有是否使用不确定性调整介入行为。

## 文件结构

- `affective_v03/models.py`：标签、状态和决策数据结构。
- `affective_v03/multimodal.py`：多模态输入合同。
- `affective_v03/uncertainty.py`：不确定性与校准指标。
- `affective_v03/fusion.py`：透明 late fusion。
- `affective_v03/support.py`：ESConv 二级支持动作选择。
- `affective_v03/policy.py`：实验 baseline 与 uncertainty-aware policy。
- `affective_v03/realizer.py`：受控中文回应。
- `affective_v03/pipeline.py`：完整决策管线。
- `affective_v03/session.py`：schema v3 实验记录。
- `tests/`：架构、策略隔离、校准和日志测试。

## 运行

```powershell
python main.py                  # v4.1 中文数值模拟测试面板
python main_v4.py               # 同上，显式版本入口
python main_v3.py               # 保留的 v0.3 实验界面
python smoke_demo.py            # 无界面冒烟验证
python experiment_runner.py     # 双条件批量演示/输入运行
python calibration_runner.py data/v0_3_experiment.json
python dataset_validator.py normalized_dataset.json
python -m unittest discover -s tests -v
```

仅使用 Python 标准库，不会在线学习或根据正式实验反馈更改 policy。

`affective_v03/scenarios.py` 中的演示输入只用于确认代码管线能够运行，不能作为研究数据、
阈值来源或实验结果。正式研究必须接入官方数据集或经过伦理流程采集的真实数据，并保存
数据集版本、标签来源和划分信息。
