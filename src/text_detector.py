"""
Full-page fine-grained text detection using PaddleOCR pretrained PP-OCRv5_server_det.

Serves as Source 2 in the Two-Source Region Proposal Architecture:
Provides millimeter-accurate text line/word geometry across the full uncropped page,
enabling independent discovery of headers, footers, side commentary columns, and filler.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger('manuscript-detector')


class FullPageTextDetector:
    """
    Pretrained full-page text detector using PP-OCRv5_server_det.
    Safely executes on CPU (bypassing oneDNN attribute incompatibility via switch_ir_optim)
    or GPU, returning bounding boxes and scores for all detected text lines.
    """

    def __init__(
        self,
        model_name: str = 'PP-OCRv5_mobile_det',
        device: str = 'auto',
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        unclip_ratio: float = 2.0,
        max_side_len: int = 960,
        enable_mkldnn: bool = True,
        cpu_threads: int = 8,
        mkldnn_cache_capacity: int = 10,
    ) -> None:
        self.model_name = model_name
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.max_side_len = max_side_len
        self.device = device
        self.enable_mkldnn = enable_mkldnn
        self.cpu_threads = cpu_threads
        self.mkldnn_cache_capacity = mkldnn_cache_capacity
        self.profiler = None
        self._predictor = None
        self._resizer = None
        self._post_op = None
        self._available = False
        self._used_mkldnn = False

        self._initialize()

    def _create_cpu_predictor(self, json_file: str, params_file: str) -> Tuple[Any, bool]:
        import paddle.inference as p_inf
        if self.enable_mkldnn:
            try:
                cfg = p_inf.Config(json_file, params_file)
                cfg.disable_gpu()
                cfg.enable_mkldnn()
                cfg.set_mkldnn_cache_capacity(self.mkldnn_cache_capacity)
                cfg.set_cpu_math_library_num_threads(self.cpu_threads)
                cfg.switch_ir_optim(False)
                pred = p_inf.create_predictor(cfg)
                # Verify PIR oneDNN runtime instruction support with a dummy forward pass
                dummy = np.zeros((1, 3, 64, 64), dtype=np.float32)
                inp_handle = pred.get_input_handle(pred.get_input_names()[0])
                inp_handle.copy_from_cpu(dummy)
                pred.run()
                return pred, True
            except Exception as e:
                logger.debug(
                    f"MKL-DNN not supported for {self.model_name} on current CPU runtime ({e}); "
                    f"falling back to multi-threaded CPU math ({self.cpu_threads} threads)."
                )

        # Standard CPU math predictor with configured thread pool
        cfg = p_inf.Config(json_file, params_file)
        cfg.disable_gpu()
        cfg.disable_mkldnn()
        cfg.set_cpu_math_library_num_threads(self.cpu_threads)
        cfg.switch_ir_optim(False)
        pred = p_inf.create_predictor(cfg)
        return pred, False

    def _initialize(self) -> None:
        """Initialize Paddle inference predictor for PP-OCRv5_server_det."""
        try:
            import paddle
            import paddle.inference as p_inf
            from paddlex.inference.models.text_detection.processors import (
                DBPostProcess,
                DetResizeForTest,
            )
            from paddlex.inference.utils.official_models import official_models

            # Determine device
            use_gpu = False
            if self.device == 'gpu' and paddle.is_compiled_with_cuda():
                use_gpu = True
            elif self.device == 'auto' and paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                use_gpu = True

            model_path = official_models.get_model_path(self.model_name)
            json_file = os.path.join(model_path, 'inference.json')
            params_file = os.path.join(model_path, 'inference.pdiparams')

            if not os.path.exists(json_file) or not os.path.exists(params_file):
                logger.warning(f"PP-OCRv5 model files not found at {model_path}")
                return

            if use_gpu:
                cfg = p_inf.Config(json_file, params_file)
                cfg.enable_use_gpu(100, 0)
                self._predictor = p_inf.create_predictor(cfg)
                logger.info(f"FullPageTextDetector initialized on GPU ({self.model_name})")
            else:
                self._predictor, self._used_mkldnn = self._create_cpu_predictor(json_file, params_file)
                accel_str = f"MKL-DNN (cache={self.mkldnn_cache_capacity})" if self._used_mkldnn else "Standard Math"
                logger.info(f"FullPageTextDetector initialized on CPU ({self.model_name}) with {self.cpu_threads} threads [{accel_str}]")

            self._resizer = DetResizeForTest(
                resize_long=self.max_side_len, limit_type='max', max_side_limit=4000
            )
            self._post_op = DBPostProcess(
                thresh=self.thresh,
                box_thresh=self.box_thresh,
                unclip_ratio=self.unclip_ratio,
            )
            self._available = True

        except Exception as e:
            logger.warning(
                f"FullPageTextDetector could not be initialized ({e}). "
                "Pipeline will fallback to morphological line clustering if required."
            )
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Execute full-page text line detection on a manuscript image.

        Args:
            image: Input image in BGR format (H, W, 3).

        Returns:
            List of detected text box dictionaries:
            [
                {
                    'x_min': int,
                    'y_min': int,
                    'x_max': int,
                    'y_max': int,
                    'score': float,
                    'polygon': list of [x, y] points
                },
                ...
            ]
        """
        if not self._available or self._predictor is None:
            return []

        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return []

        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_imgs, shape_infos = self._resizer([rgb_img])

        if hasattr(self, 'profiler') and self.profiler is not None:
            det_px = resized_imgs[0].shape[0] * resized_imgs[0].shape[1]
            self.profiler.record_pixels('text_detection', det_px)
            self.profiler.record_count('text_detector_calls', 1)

        # Normalize with ImageNet mean and std
        norm_img = resized_imgs[0].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (norm_img - mean) / std

        # Convert to CHW and batch format
        chw = np.transpose(norm_img, (2, 0, 1))[np.newaxis, :].astype(np.float32)

        # Inference
        inp_handle = self._predictor.get_input_handle(self._predictor.get_input_names()[0])
        inp_handle.copy_from_cpu(chw)
        self._predictor.run()

        out_handle = self._predictor.get_output_handle(self._predictor.get_output_names()[0])
        preds = out_handle.copy_to_cpu()

        if preds.ndim == 2:
            preds = preds[np.newaxis, np.newaxis, ...]
        elif preds.ndim == 3:
            preds = preds[np.newaxis, ...]

        # DBNet postprocessing
        polys, scores = self._post_op([preds], shape_infos)

        if not polys or len(polys[0]) == 0:
            return []

        boxes = []
        for poly, score in zip(polys[0], scores[0]):
            xs = poly[:, 0]
            ys = poly[:, 1]
            xmin = max(0, int(np.floor(xs.min())))
            ymin = max(0, int(np.floor(ys.min())))
            xmax = min(w, int(np.ceil(xs.max())))
            ymax = min(h, int(np.ceil(ys.max())))

            if xmax > xmin and ymax > ymin:
                boxes.append({
                    'x_min': xmin,
                    'y_min': ymin,
                    'x_max': xmax,
                    'y_max': ymax,
                    'score': float(score),
                    'polygon': poly.tolist(),
                })

        if hasattr(self, 'profiler') and self.profiler is not None:
            self.profiler.record_count('num_ocr_text_boxes', len(boxes))

        return boxes
