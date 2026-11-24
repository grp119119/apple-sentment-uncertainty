"""Test run YuNet + HSEmotion on the 3 real face photos with existing Adaptive Policy."""

import glob
import os
import sys
from pathlib import Path

# 添加当前路径至 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from affective_v03.models import AffectLabel, PolicyContext
from affective_v03.policy import UncertaintyAwarePolicy
from affective_v04.perception.adapter import record_to_evidence, safe_fuse_records
from affective_v04.perception.visual_detector import VisualEmotionDetector

detector = VisualEmotionDetector()
policy = UncertaintyAwarePolicy()

img_paths = sorted(glob.glob("data/test_faces/*.jpg"))
print(f"找到 {len(img_paths)} 张测试照片:")

for idx, img_path in enumerate(img_paths, 1):
    print(f"\n=======================================================")
    print(f"【照片 {idx}】: {os.path.basename(img_path)}")
    record = detector.analyze_image(img_path)
    
    print(f"• 人脸检出: {record.available}")
    print(f"• 信号质量 (人脸置信度): {record.signal_quality:.4f}")
    print(f"• 推理耗时: {record.latency_ms:.2f} ms")
    
    if not record.available:
        print(f"• 缺失原因: {record.missing_reason}")
        continue
    
    print("\n【HSEmotion 8 类原始输出】:")
    for k, v in record.raw_scores.items():
        print(f"  - {k:12s}: {v:.4f}")
        
    print("\n【映射后 Ekman 7 类分布】:")
    for label in AffectLabel:
        val = record.mapped_scores.get(label, 0.0)
        print(f"  - {label.value:10s}: {val:.4f}")
        
    # 送入安全融合与决策
    fusion = safe_fuse_records([record])
    dominant = fusion.affect.dominant_emotion.value
    print(f"\n【融合结果】:")
    print(f"  - 主导情绪: {dominant} (Valence: {fusion.affect.valence:.2f}, Arousal: {fusion.affect.arousal:.2f})")
    print(f"  - 不确定性熵: {fusion.uncertainty.predictive_entropy:.4f}")
    print(f"  - Top-2 Margin: {fusion.uncertainty.top_two_margin:.4f}")
    
    # 驱动既有 Adaptive Policy 决策
    context = PolicyContext(
        user_initiated=True,
    )
    decision = policy.decide(fusion.affect, fusion.uncertainty, context)
    print(f"\n【Adaptive Policy 决策产出】:")
    print(f"  - 介入模式 (Mode): {decision.mode.value}")
    print(f"  - 介入强度 (Intensity): {decision.intensity.value}")
    print(f"  - 支持动作 (SupportAction): {decision.support_action.value}")
    print(f"  - 认识论姿态: {decision.epistemic_style.value}")
    print(f"  - 决策理由: {decision.reasons}")
