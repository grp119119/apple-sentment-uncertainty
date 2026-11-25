"""Local visual emotion detector using OpenCV YuNet + HSEmotionONNX."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import onnxruntime as ort

from affective.models import AffectLabel
from .adapter import HSEMOTION_8_MAP, map_raw_scores_to_ekman
from .contracts import (
    PerceptionModality,
    PerceptionRecord,
    PerceptionSource,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_YUNET_PATH = ROOT / "models" / "visual" / "face_detection_yunet_2023mar.onnx"
DEFAULT_HSEMOTION_PATH = ROOT / "models" / "visual" / "enet_b0_8_best_afew.onnx"

HSEMOTION_CLASSES = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutrality",
    "Sadness",
    "Surprise",
]


class VisualEmotionDetector:
    """Offline / local face detection and emotion recognition pipeline."""

    def __init__(
        self,
        yunet_model_path: str | Path | None = None,
        fer_model_path: str | Path | None = None,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ) -> None:
        self.yunet_path = Path(yunet_model_path or DEFAULT_YUNET_PATH)
        self.fer_path = Path(fer_model_path or DEFAULT_HSEMOTION_PATH)

        if not self.yunet_path.exists():
            raise FileNotFoundError(f"YuNet 模型权重未找到: {self.yunet_path}")
        if not self.fer_path.exists():
            raise FileNotFoundError(f"HSEmotion 模型权重未找到: {self.fer_path}")

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.img_size = 224

        # 初始化 YuNet (尺寸在推理时自适应调整)
        self.detector = cv2.FaceDetectorYN.create(
            model=str(self.yunet_path),
            config="",
            input_size=(320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000,
        )

        # 优先使用可用执行提供器 (如 CUDA 或 CPU)
        available_providers = ort.get_available_providers()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]

        # 直接加载本地 ONNX
        self.ort_session = ort.InferenceSession(str(self.fer_path), providers=providers)
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(HSEMOTION_CLASSES)}

    def _preprocess_face(self, img: np.ndarray) -> np.ndarray:
        # BGR 格式输入预处理 (ImageNet 均值方差归一化)
        x = cv2.resize(img, (self.img_size, self.img_size)) / 255.0
        x[..., 0] = (x[..., 0] - 0.485) / 0.229
        x[..., 1] = (x[..., 1] - 0.456) / 0.224
        x[..., 2] = (x[..., 2] - 0.406) / 0.225
        return x.transpose(2, 0, 1).astype("float32")[np.newaxis, ...]

    def analyze_image(
        self,
        image_input: str | Path | np.ndarray,
        source: PerceptionSource = PerceptionSource.OFFLINE_DATASET,
    ) -> PerceptionRecord:
        """Detect face and predict facial emotion distribution from a file or array."""
        start_time = time.perf_counter()
        now = time.time()

        # 加载图片
        if isinstance(image_input, (str, Path)):
            img_path = str(image_input)
            if not os.path.exists(img_path):
                return PerceptionRecord(
                    modality=PerceptionModality.VISUAL,
                    source=source,
                    model_name="YuNet+HSEmotionONNX",
                    model_version="enet_b0_8_best_afew",
                    timestamp=now,
                    raw_scores={},
                    mapped_scores={},
                    signal_quality=0.0,
                    available=False,
                    missing_reason=f"FILE_NOT_FOUND: {img_path}",
                    latency_ms=0.0,
                )
            img = cv2.imread(img_path)
            if img is None:
                return PerceptionRecord(
                    modality=PerceptionModality.VISUAL,
                    source=source,
                    model_name="YuNet+HSEmotionONNX",
                    model_version="enet_b0_8_best_afew",
                    timestamp=now,
                    raw_scores={},
                    mapped_scores={},
                    signal_quality=0.0,
                    available=False,
                    missing_reason=f"INVALID_IMAGE_DATA: {img_path}",
                    latency_ms=0.0,
                )
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise TypeError("image_input 必须是文件路径或 numpy ndarray")

        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))

        # 1. YuNet 人脸检测
        _, faces = self.detector.detect(img)

        # 2. 人脸未检出处理 (平稳降级)
        if faces is None or len(faces) == 0:
            latency = (time.perf_counter() - start_time) * 1000
            return PerceptionRecord(
                modality=PerceptionModality.VISUAL,
                source=source,
                model_name="YuNet+HSEmotionONNX",
                model_version="enet_b0_8_best_afew",
                timestamp=now,
                raw_scores={},
                mapped_scores={},
                signal_quality=0.0,
                available=False,
                missing_reason="NO_FACE_DETECTED",
                latency_ms=latency,
            )

        # 3. 多人场景选择最大的人脸 (最接近主体的核心参与者)
        best_face = faces[0]
        max_area = 0
        for face in faces:
            area = face[2] * face[3]
            if area > max_area:
                max_area = area
                best_face = face

        # 提取人脸边界与置信度 (置信度为 face[14])
        x, y, fw, fh = [int(v) for v in best_face[:4]]
        confidence = float(best_face[14])

        # 坐标安全裁剪
        x = max(0, x)
        y = max(0, y)
        fw = min(fw, w - x)
        fh = min(fh, h - y)

        if fw <= 10 or fh <= 10:
            latency = (time.perf_counter() - start_time) * 1000
            return PerceptionRecord(
                modality=PerceptionModality.VISUAL,
                source=source,
                model_name="YuNet+HSEmotionONNX",
                model_version="enet_b0_8_best_afew",
                timestamp=now,
                raw_scores={},
                mapped_scores={},
                signal_quality=0.0,
                available=False,
                missing_reason="FACE_TOO_SMALL",
                latency_ms=latency,
            )

        face_crop = img[y : y + fh, x : x + fw]

        # 4. HSEmotion 表情推理
        inp = self._preprocess_face(face_crop)
        ort_out = self.ort_session.run(None, {"input": inp})[0][0]

        # Softmax 归一化概率
        exp_vals = np.exp(ort_out - np.max(ort_out))
        probs = exp_vals / np.sum(exp_vals)

        raw_dict = {
            self.idx_to_class[i]: float(probs[i]) for i in range(len(probs))
        }
        # 记录检测框供前端实时可视化绘制
        raw_dict["_bbox_x"] = float(x)
        raw_dict["_bbox_y"] = float(y)
        raw_dict["_bbox_w"] = float(fw)
        raw_dict["_bbox_h"] = float(fh)

        # 5. 映射为系统标准的 Ekman 7 类
        mapped_ekman = map_raw_scores_to_ekman(raw_dict, HSEMOTION_8_MAP)

        latency = (time.perf_counter() - start_time) * 1000

        return PerceptionRecord(
            modality=PerceptionModality.VISUAL,
            source=source,
            model_name="YuNet+HSEmotionONNX",
            model_version="enet_b0_8_best_afew",
            timestamp=now,
            raw_scores=raw_dict,
            mapped_scores=mapped_ekman,
            signal_quality=min(1.0, max(0.0, confidence)),
            available=True,
            latency_ms=latency,
        )
