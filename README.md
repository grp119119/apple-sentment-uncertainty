# 根据情绪判断提供支持的交互原型 v4.2

这个原型用摄像头和麦克风估计用户的情绪，再决定是直接回应、先问一句，还是等一等。它也可以用手动输入的数值模拟同样的过程，方便比较两套策略在同一输入下的表现。

系统先按信号质量合并各路输入，估计七类情绪的概率（Ekman 六类情绪加中性），再结合数据是否充足、音视频判断是否一致等信息，决定回应的时机、语气和强度。回应使用中文模板，支持方式来自 ESConv 的八个类别，例如提问、安慰和建议。实时面板可在结束后导出会话记录。

## 怎么运行

建议使用 Python 3.10–3.12。在项目根目录运行对应的命令。

```powershell
python main_v4_manual_sim.py      # 手动输入数值，查看系统怎么回应，无需摄像头或麦克风
python experiment_runner.py --all  # 用内置示例比较两套策略
```

要用摄像头和麦克风，先安装依赖，再选择图形界面或命令行入口。

```powershell
python -m pip install -r requirements.txt
python main_v4.py               # 实时图形界面
python run_live_interaction.py  # 实时命令行界面
python debug_audio.py           # 排查麦克风和语音识别问题
```

处理自己的数据时，把下面的示例文件名换成你的文件。

```powershell
python experiment_runner.py --input observation.json  # 输入 JSON 须包含 evidence 数组
python calibration_runner.py session.json             # 从 v3 格式的实验记录生成统计报告
python dataset_validator.py dataset.json               # 检查整理后的数据标签
```

安装依赖后，可用 `python -m unittest discover -s tests -v` 运行测试。

## 两套策略怎么比较

两套策略用同样的方法估计情绪、选择支持方式和生成回应。用户要求暂停、当前不愿接受支持，或安全规则阻止时，两套策略都会暂缓（`DEFER`）。其余情况下，两套策略对判断不确定性的处理不同。

- 仅看情绪（`EmotionOnlyPolicy`）直接回应（`DIRECT`）。
- 考虑不确定性（`UncertaintyAwarePolicy`）在信号太差、数据不足或判断很模糊时，用户主动发起就先问一句（`CONFIRM`），否则等待（`WAIT`）。如果判断仍有些不确定，或音视频的判断明显不一致，就用试探的语气回应（`TENTATIVE`）。证据充足、判断清楚时直接回应（`DIRECT`）。

核心流程在 [affective/pipeline.py](affective/pipeline.py)，两套策略在 [affective/policy.py](affective/policy.py)，实时音视频处理在 [affective/perception/](affective/perception/)。

## 正式实验前

当前阈值是尚未验证的默认值（`PILOT_DRAFT_NOT_VALIDATED`），正式实验前需要先校准并固定。衡量音视频判断分歧的指标仍处于试验阶段。内置示例只用于检查流程能否跑通。
