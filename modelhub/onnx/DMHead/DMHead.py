"""
DMHead - 360° 6D 头部姿态估计（WHENet + 6DRepNet 双模型融合）
来源: https://github.com/PINTO0309/DMHead

模型输入:  float32 [N,3,224,224]，RGB，0~255（前处理已融合进模型，不要再做归一化）
模型输出:  float32 [N,3]，顺序为 [Yaw, Roll, Pitch]，单位为角度
"""
from pathlib import Path
from typing import List, Sequence, Union

import cv2
import numpy as np

from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)


class DMHead:
    VARIANTS = ('nomask', 'mask')
    MODEL_FILES = {
        'nomask': 'dmhead_nomask_Nx3x224x224.onnx',
        'mask':   'dmhead_mask_Nx3x224x224.onnx',
    }

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    @staticmethod
    def get_model_path(variant: str = 'nomask') -> Path:
        if variant not in DMHead.MODEL_FILES:
            raise ValueError(f"unknown DMHead variant '{variant}', expected one of {DMHead.VARIANTS}")
        return Path(__file__).parent / DMHead.MODEL_FILES[variant]

    @staticmethod
    def is_model_available(variant: str = 'nomask') -> bool:
        return DMHead.get_model_path(variant).exists()

    def __init__(self, device_info: ORTDeviceInfo, variant: str = 'nomask'):
        if device_info not in DMHead.get_available_devices():
            raise Exception(f'device_info {device_info} is not in available devices for DMHead')

        path = DMHead.get_model_path(variant)
        if not path.exists():
            raise FileNotFoundError(
                f'{path} not found. 请从 https://github.com/PINTO0309/DMHead/releases 下载 '
                f'{path.name} 并放置到 {path.parent}'
            )

        self.variant = variant
        self._sess = sess = InferenceSession_with_device(str(path), device_info)
        self._input_name = sess.get_inputs()[0].name

        input_shape = sess.get_inputs()[0].shape
        self._input_height = int(input_shape[2]) if isinstance(input_shape[2], int) else 224
        self._input_width = int(input_shape[3]) if isinstance(input_shape[3], int) else 224

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """HWC BGR (uint8/float32) -> CHW RGB float32 [0,255]"""
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = img[..., :3]

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8) if img.max() > 1.0 else (img * 255).astype(np.uint8)

        if img.shape[0] != self._input_height or img.shape[1] != self._input_width:
            img = cv2.resize(img, (self._input_width, self._input_height), interpolation=cv2.INTER_AREA)

        rgb = img[..., ::-1]
        chw = rgb.transpose(2, 0, 1)
        return np.ascontiguousarray(chw, dtype=np.float32)

    def extract(self, img: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
        """
        img: np.ndarray HWC(BGR) 或 list[np.ndarray]
        returns (N,3) float32 [pitch, yaw, roll] 角度制；yaw 范围 -180~180
        """
        if isinstance(img, np.ndarray) and img.ndim == 3:
            imgs = [img]
        else:
            imgs = list(img)

        nchw = np.stack([self._preprocess(im) for im in imgs], axis=0)

        # 模型原始输出顺序: [yaw, roll, pitch]
        out = self._sess.run(None, {self._input_name: nchw})[0]
        out = np.asarray(out, dtype=np.float32).reshape(-1, 3)

        yaw, roll, pitch = out[:, 0], out[:, 1], out[:, 2]
        return np.stack([pitch, yaw, roll], axis=1)

    def extract_single(self, img: np.ndarray):
        """returns (pitch, yaw, roll) 角度制 float"""
        pitch, yaw, roll = self.extract(img)[0]
        return float(pitch), float(yaw), float(roll)
