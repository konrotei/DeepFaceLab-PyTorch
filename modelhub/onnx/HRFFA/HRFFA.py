"""HRFFA: High-Angle Robust Fast FaceAlignment (68 点, iBUG 顺序).

来源: https://github.com/PINTO0309/High-Angle_Robust_Fast_FaceAlignment
权重: https://github.com/PINTO0309/High-Angle_Robust_Fast_FaceAlignment/releases/tag/weights

特点:
  - 对极端头部姿态鲁棒: yaw ±90°(全侧脸), pitch ±85°(俯仰), roll 0~360°(平面旋转)
  - 输出 68 点 (ibug68), 与 DFL 现有 68 点流程 (get_transform_mat / XSeg 等) 完全兼容
  - 额外输出每个点的 3 类可见性 (0=画面外 / 1=遮挡 / 2=可见), 存于 self.last_visibility

ONNX I/O 约定 (与上游一致):
  input   images      float32 [1, 3, S, S]   RGB, 归一化见 INPUT_NORMS
  output  points      float32 [1, 68, 2]     裁切内相对坐标 (0..1, 可越界)
  output  vis_logits  float32 [1, 68, 3]

注意: 该模型在 **整头部裁切** 上训练(不是紧凑人脸框), 裁切几何为
  "头部框长边 × (1 + 2·pad), pad=0.05, 相似变换到 S×S".
DFL 的检测器给的是人脸框, 因此 Extractor 需要用更大的边距并向上偏移来近似头部框,
这通过类属性 CROP_MARGIN_RATIO / CROP_SHIFT_UP_RATIO 告知 extract_landmarks().
"""
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)


# 变体名 -> (onnx 文件名, 输入归一化)
VARIANTS = {
    # 推荐: 精度/速度平衡 (9.0M params, CPU ~12ms)
    'vitt-256': ('hrffa_vitt_ibug68_1x3x256x256.onnx', 'center05'),
    # 极轻量 CNN (1.6M params, CPU ~5ms), 精度略低
    'hg0-256':  ('hrffa_hg0_ibug68_1x3x256x256.onnx',  'center05'),
    # 96px 低分辩率版本, 仅适合极低算力场景
    'vitt-96':  ('hrffa_vitt_ibug68_1x3x96x96.onnx',   'center05'),
    'hg0-96':   ('hrffa_hg0_ibug68_1x3x96x96.onnx',    'center05'),
    # 教师模型 (308M params, 1.2GB), 精度最高但很慢, 仅 GPU
    'vitl-320': ('hrffa_vitl_ibug68_1x3x320x320.onnx', 'imagenet'),
}

INPUT_NORMS = {
    'imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'center05': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}

RELEASE_BASE_URL = ('https://github.com/PINTO0309/High-Angle_Robust_Fast_FaceAlignment'
                    '/releases/download/weights/')


class HRFFA:
    """
    HRFFA 68 点特征点标记器 (ONNX Runtime).

    接口与 modelhub 其他 landmarker 一致:
        extract(img_bgr) -> [np.ndarray (68, 2)]   坐标相对于传入 img
    """

    # 供 Extractor.extract_landmarks() 读取的裁切提示:
    # 人脸框每边扩大 30% 宽度, 并整体上移 12% 高度, 以近似训练时用的整头部框
    CROP_MARGIN_RATIO = 0.30
    CROP_SHIFT_UP_RATIO = 0.12
    # 与上游训练/评估一致的头部框 padding
    CROP_PAD = 0.05

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo, variant: str = 'vitt-256',
                 auto_download: bool = True):
        if variant not in VARIANTS:
            raise ValueError(f'未知 HRFFA 变体: {variant}, 可选: {list(VARIANTS.keys())}')
        self.variant = variant
        filename, norm = VARIANTS[variant]
        path = Path(__file__).parent / filename

        if not path.exists() and auto_download:
            self._download(filename, path)
        if not path.exists():
            raise FileNotFoundError(
                f'HRFFA 模型缺失: {path}\n'
                f'请从 {RELEASE_BASE_URL}{filename} 下载并放入 {path.parent}')

        self._sess = InferenceSession_with_device(str(path), device_info)
        inp = self._sess.get_inputs()[0]
        self._input_name = inp.name
        shape = inp.shape
        if len(shape) != 4 or not isinstance(shape[2], int) or shape[2] != shape[3]:
            raise ValueError(f'HRFFA 输入应为 [N,3,S,S], 实际 {shape}')
        self.out_size = int(shape[2])

        out_names = [o.name for o in self._sess.get_outputs()]
        for n in ('points', 'vis_logits'):
            if n not in out_names:
                raise ValueError(f'HRFFA 模型缺少输出 `{n}`, 实际 {out_names}')

        mean, std = INPUT_NORMS[norm]
        self._mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

        # ── TRT BF16 加速 (与其他 modelhub 模型一致的可选路径) ──
        _trt_path = None
        try:
            from xlib.trt import find_trt_engine
            _trt_path = find_trt_engine(str(path), path.stem)
        except Exception:
            pass
        if _trt_path:
            try:
                from xlib.trt import TRTInferenceSession
                self._sess = TRTInferenceSession(_trt_path)
            except Exception as e:
                import warnings as _w
                _w.warn(f'TRT fallback: {e}')

        # 最近一次 extract 的可见性 (68,), 0=画面外 / 1=遮挡 / 2=可见
        self.last_visibility: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _download(filename: str, dst: Path) -> None:
        url = RELEASE_BASE_URL + filename
        try:
            import urllib.request
            print(f'[HRFFA] 模型不存在, 正在从 GitHub 下载: {url}')
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix('.onnx.part')
            urllib.request.urlretrieve(url, str(tmp))
            tmp.rename(dst)
            print(f'[HRFFA] 下载完成: {dst}')
        except Exception as e:
            print(f'[HRFFA] 自动下载失败 ({e}), 请手动下载: {url}')

    # ------------------------------------------------------------------
    def _crop_transform(self, w: int, h: int) -> np.ndarray:
        """把整张传入图(视作头部框)按上游 crop_affine 几何映射到 S×S 的 3x3 矩阵."""
        cx, cy = w / 2.0, h / 2.0
        side = max(w, h) * (1.0 + 2.0 * self.CROP_PAD)
        s = self.out_size / side
        half = self.out_size / 2.0
        return np.array([[s, 0.0, half - s * cx],
                         [0.0, s, half - s * cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def extract(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Args:
            img: BGR 头部区域图像 (任意尺寸). 整张图被视作头部框.
        Returns:
            [landmarks] , landmarks 为 (68, 2) float32, 坐标相对于 img.
        """
        if img is None or img.size == 0:
            return []
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        h, w = img.shape[:2]
        T = self._crop_transform(w, h)
        crop = cv2.warpAffine(img, T[:2], (self.out_size, self.out_size),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0
        tensor = ((rgb - self._mean) / self._std)[None].astype(np.float32)

        points, vis_logits = self._sess.run(['points', 'vis_logits'], {self._input_name: tensor})
        pts = points[0].astype(np.float64) * self.out_size                 # 裁切像素坐标
        homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        img_xy = (np.linalg.inv(T) @ homo.T).T[:, :2]                       # 传入图坐标

        self.last_visibility = vis_logits[0].argmax(axis=1).astype(np.int64)
        return [img_xy.astype(np.float32)]