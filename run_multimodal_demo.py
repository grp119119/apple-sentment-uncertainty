"""End-to-End Real Multimodal Perception & Uncertainty-Aware Policy Demo.

Combines:
  1. Real Face Photo (YuNet + HSEmotion) -> Visual Ekman Distribution
  2. Real Audio Recording (SenseVoiceSmall ASR + emotion2vec+ SER) -> Acoustic Ekman + ASR Transcript
  3. Safe Late Fusion (Multimodal Observation & Uncertainty Profiling)
  4. Uncertainty-Aware JITAI Policy (Adaptive Intervention Decision)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from affective_v03.models import AffectLabel, UserNeed
from affective_v03.policy import PolicyContext, UncertaintyAwarePolicy
from affective_v04.perception.adapter import safe_fuse_records
from affective_v04.perception.audio_detector import AudioEmotionDetector
from affective_v04.perception.visual_detector import VisualEmotionDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Multimodal Real Perception & Policy Demo")
    parser.add_argument(
        "--image",
        type=str,
        default="data/test_faces/96e7055064fe4d96722915c85412cdbb.jpg",
        help="Path to face image file",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="data/test_audio/北京工业大学中蓝公寓.m4a",
        help="Path to audio file (.wav, .m4a, etc.)",
    )
    parser.add_argument(
        "--need",
        type=str,
        default="COMFORT",
        choices=[n.name for n in UserNeed],
        help="Explicit user need (default: COMFORT)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    audio_path = Path(args.audio)
    user_need = UserNeed[args.need]

    print("=" * 68)
    print(" 🎓 空间人因多模态情感感知与自适应干预策略 (End-to-End Multimodal Demo)")
    print("=" * 68)

    # 1. 视觉感知
    print(f"\n[1/4] 📷 视觉感知 (Face Detection & Emotion) -> {image_path.name}")
    visual_detector = VisualEmotionDetector()
    vis_rec = visual_detector.analyze_image(image_path)
    if vis_rec.available:
        top_vis = max(vis_rec.mapped_scores.items(), key=lambda x: x[1])
        print(f"  • 人脸置信度 (Signal Quality): {vis_rec.signal_quality:.2%}")
        print(f"  • 主导表情 (Top Emotion):      {top_vis[0].value.upper()} ({top_vis[1]:.2%})")
        print(f"  • 视觉感知耗时:               {vis_rec.latency_ms:.1f} ms")
    else:
        print(f"  • 未检测到人脸: {vis_rec.missing_reason}")

    # 2. 听觉/文本感知
    print(f"\n[2/4] 🎙️ 语音感知 (Dual-stream ASR + SER) -> {audio_path.name}")
    audio_detector = AudioEmotionDetector()
    audio_rec, text_rec = audio_detector.analyze_audio(audio_path)
    if audio_rec.available:
        top_aud = max(audio_rec.mapped_scores.items(), key=lambda x: x[1])
        print(f"  • 声学情绪 (Top SER):          {top_aud[0].value.upper()} ({top_aud[1]:.2%})")
        print(f"  • 语音转写 (ASR Transcript):   \"{text_rec.transcript}\"")
        print(f"  • 音频感知耗时:               {audio_rec.latency_ms:.1f} ms")
    else:
        print(f"  • 音频静音或不可用: {audio_rec.missing_reason}")

    # 3. 多模态后期融合与不确定性画像
    print(f"\n[3/4] 🧠 多模态后期融合与不确定性画像 (Late Fusion & Uncertainty Profiling)")
    active_records = [r for r in [vis_rec, audio_rec] if r.available]
    fusion_result = safe_fuse_records(active_records)
    aff = fusion_result.affect
    unc = fusion_result.uncertainty

    disagreement_str = (
        f"{unc.cross_modal_disagreement:.4f}"
        if unc.cross_modal_disagreement is not None
        else "N/A (单模态)"
    )
    print(f"  • 跨模态冲突度 (Cross-Modal Disagreement): {disagreement_str}")
    print(f"  • 预测信息熵 (Predictive Entropy):          {unc.predictive_entropy:.4f}")
    print(f"  • 优势边际 (Top-Two Margin):               {unc.top_two_margin:.4f}")
    print(f"  • 综合信号质量 (Overall Signal Quality):    {unc.signal_quality:.2%}")
    top_fused = sorted(aff.distribution.items(), key=lambda x: x[1], reverse=True)[:3]
    fused_str = ", ".join([f"{k.value}: {v:.1%}" for k, v in top_fused if v > 0.01])
    print(f"  • 融合后情绪分布 (Top Fused Distribution): {fused_str}")

    # 4. Uncertainty-Aware Policy 决策
    print(f"\n[4/4] 🛡️ 确定性自适应策略响应 (Uncertainty-Aware Adaptive Policy)")
    policy = UncertaintyAwarePolicy()
    context = PolicyContext(user_initiated=True, user_need=user_need)
    decision = policy.decide(aff, unc, context)

    print(f"  • 决策干预模式 (Intervention Mode):  【 {decision.mode.value.upper()} 】")
    print(f"  • 干预强度 (Intensity):              {decision.intensity.value.upper()}")
    print(f"  • 推荐支持动作 (Support Action):     {decision.support_action.value}")
    print(f"  • 认知风格 (Epistemic Style):        {decision.epistemic_style.value}")
    print(f"  • 判定依据 (Decision Reasons):")
    for r in decision.reasons:
        print(f"      - {r}")

    print("\n" + "=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())

