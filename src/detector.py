"""
PP-DocLayoutV2 Hybrid Detection Engine for Manuscript Layout Analysis.

Architecture:
    Layer 1 — PP-DocLayoutV2 pretrained layout detection via PaddleOCR
    Layer 2 — Assignment-specific class mapping
    Layer 3 — Manuscript-aware heuristic + OpenCV filler refinement
"""

import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple

# Suppress low-level GLog notices
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2
import numpy as np

from src.utils import suppress_c_output, resolve_device
from src.text_detector import FullPageTextDetector
from src.region_reconstructor import RegionReconstructor
from src.postprocessor import _iou

logger = logging.getLogger('manuscript-detector')


# Global cache for heavy neural network predictors to avoid redundant re-initialization
_MODEL_CACHE: Dict[str, Any] = {}


class ManuscriptDetector:
    """
    Hybrid manuscript layout region detector built on PP-DocLayoutV2.
    """
    profiler: Any = None
    enable_mkldnn: bool = True
    cpu_threads: int = 8
    mkldnn_cache_capacity: int = 10
    opencv_threads: int = 2
    warmup_enabled: bool = False

    def __init__(self, config: Optional[dict] = None,
                 class_mapping: Optional[dict] = None,
                 device: Optional[str] = None):
        self.config = config or {}
        self.class_mapping = class_mapping or {}
        self.requested_device = device or self.config.get('model', {}).get('device', 'auto')
        self.device = resolve_device(self.requested_device)
        logger.debug(f"ManuscriptDetector device resolved: {self.device} (requested: {self.requested_device})")

        # Resolve CPU performance configuration
        perf_cfg = self.config.get('performance', {})
        self.enable_mkldnn = perf_cfg.get('enable_mkldnn', True)
        self.mkldnn_cache_capacity = perf_cfg.get('mkldnn_cache_capacity', 10)
        self.opencv_threads = perf_cfg.get('opencv_threads', 2)
        self.warmup_enabled = perf_cfg.get('warmup', True)

        from src.utils import resolve_cpu_threads, configure_cpu_environment
        raw_threads = perf_cfg.get('cpu_threads', 'auto')
        self.cpu_threads = resolve_cpu_threads(raw_threads)

        # Apply anti-oversubscription controls across OpenCV, OpenMP, MKL
        configure_cpu_environment(self.cpu_threads, opencv_threads=self.opencv_threads)

        startup_msg = (
            f"[Performance] Selected CPU threads: {self.cpu_threads} | "
            f"enable_mkldnn: {self.enable_mkldnn} | "
            f"cache_capacity: {self.mkldnn_cache_capacity} | "
            f"opencv_threads: {self.opencv_threads}"
        )
        logger.debug(startup_msg)

        self.model = None
        self.model_available = False

        heur = self.config.get('spatial_heuristics', {})
        self.header_zone = heur.get('header_zone', 0.18)
        self.footer_zone = heur.get('footer_zone', 0.82)
        self.side_margin = heur.get('side_margin', 0.15)

        self.direct_map = self.class_mapping.get('direct_mappings', {
            'header': 'header', 'footer': 'footer', 'text': 'main_text',
        })
        self.candidate_map = self.class_mapping.get('candidate_mappings', {
            'paragraph_title': 'header', 'number': 'header',
            'aside_text': 'side_text', 'reference': 'side_text',
            'reference_content': 'side_text', 'footnote': 'side_text',
            'vision_footnote': 'side_text',
            'image': 'filler', 'seal': 'filler', 'chart': 'filler',
            'formula_number': 'filler', 'inline_formula': 'filler',
            'table': 'main_text', 'vertical_text': 'main_text',
        })
        self.ignore_labels = set(
            self.class_mapping.get('ignore_labels', [
                'abstract', 'algorithm', 'footer_image', 'header_image',
                'display_formula', 'figure_title', 'doc_title', 'content',
            ])
        )

        self.recognizer = None
        self._load_model()
        from src.semantic_refiner import SemanticRefiner
        self.refiner = SemanticRefiner(recognizer=self.recognizer, config=self.config)

        if self.warmup_enabled and self.device == 'cpu':
            self._warmup()

    def _warmup(self) -> None:
        """Run warm-up passes through models so JIT/PIR execution graphs stabilize."""
        dummy = np.full((128, 256, 3), 200, dtype=np.uint8)
        try:
            if self.model_available and self.model is not None:
                _ = list(self.model.predict(dummy))
            if self.text_detector is not None and self.text_detector.is_available:
                _ = self.text_detector.detect(dummy)
            if self.recognizer is not None:
                _ = list(self.recognizer.predict(dummy[20:60, 20:100]))
            logger.debug("CPU model warmup complete.")
        except Exception as e:
            logger.debug(f"Warmup warning (non-fatal): {e}")

    # Model loading

    def _load_model(self) -> None:
        """Load PP-DocLayoutV2 via PaddleOCR LayoutDetection and TextRecognition with caching."""
        model_cfg = self.config.get('model', {})
        model_name = model_cfg.get('name', 'PP-DocLayoutV2')
        threshold = model_cfg.get('confidence_threshold', 0.15)

        layout_key = f"layout_{model_name}_{self.device}_{self.cpu_threads}_{self.enable_mkldnn}"
        if layout_key in _MODEL_CACHE:
            self.model = _MODEL_CACHE[layout_key]
            self.model_available = True
            logger.debug(f"PP-DocLayoutV2 retrieved from cache ({layout_key})")
        else:
            model_kwargs = {
                'model_name': model_name,
                'threshold': threshold,
                'device': self.device,
            }
            if self.device == 'cpu':
                model_kwargs['enable_mkldnn'] = self.enable_mkldnn
                model_kwargs['cpu_threads'] = self.cpu_threads
                model_kwargs['mkldnn_cache_capacity'] = self.mkldnn_cache_capacity

            try:
                with suppress_c_output():
                    from paddleocr import LayoutDetection
                    self.model = LayoutDetection(**model_kwargs)
                self.model_available = True
                _MODEL_CACHE[layout_key] = self.model
                logger.debug(
                    f"PP-DocLayoutV2 loaded (threshold={threshold}, device={self.device}, "
                    f"mkldnn={self.enable_mkldnn}, threads={self.cpu_threads})"
                )
            except ImportError:
                logger.warning("paddleocr not installed.")
            except Exception as e:
                logger.warning(f"PP-DocLayoutV2 load failed: {e}.")

        # Load TextRecognition model for multilingual (Sanskrit/Devanagari + English) script inspection
        rec_key = f"rec_{self.device}_{self.cpu_threads}_{self.enable_mkldnn}"
        if rec_key in _MODEL_CACHE:
            self.recognizer = _MODEL_CACHE[rec_key]
            logger.debug(f"TextRecognition retrieved from cache ({rec_key})")
        else:
            rec_kwargs = {
                'device': self.device,
            }
            if self.device == 'cpu':
                rec_kwargs['enable_mkldnn'] = self.enable_mkldnn
                rec_kwargs['cpu_threads'] = self.cpu_threads
                rec_kwargs['mkldnn_cache_capacity'] = self.mkldnn_cache_capacity

            try:
                with suppress_c_output():
                    from paddleocr import TextRecognition
                    try:
                        self.recognizer = TextRecognition(model_name='devanagari_PP-OCRv5_mobile_rec', **rec_kwargs)
                        logger.debug(f"devanagari_PP-OCRv5_mobile_rec loaded for Indic/Sanskrit script inspection (device={self.device})")
                    except Exception:
                        self.recognizer = TextRecognition(model_name='PP-OCRv6_medium_rec', **rec_kwargs)
                        logger.debug(f"PP-OCRv6_medium_rec loaded for second-stage line inspection (device={self.device})")
                _MODEL_CACHE[rec_key] = self.recognizer
            except Exception as e:
                logger.warning(f"TextRecognition load failed: {e}. Running with heuristic script analysis.")
                self.recognizer = None

        # Source 2: Full-Page fine-grained text line detector (PP-OCRv5_mobile_det)
        det_cfg = self.config.get('text_detection', {})
        det_model_name = det_cfg.get('model_name', 'PP-OCRv5_mobile_det')
        det_key = f"det_{det_model_name}_{self.device}_{self.cpu_threads}_{self.enable_mkldnn}"
        if det_key in _MODEL_CACHE:
            self.text_detector = _MODEL_CACHE[det_key]
            logger.debug(f"FullPageTextDetector retrieved from cache ({det_key})")
        else:
            try:
                self.text_detector = FullPageTextDetector(
                    model_name=det_model_name,
                    device=self.device,
                    thresh=det_cfg.get('thresh', 0.3),
                    box_thresh=det_cfg.get('box_thresh', 0.6),
                    unclip_ratio=det_cfg.get('unclip_ratio', 2.0),
                    max_side_len=det_cfg.get('max_side_len', 960),
                    enable_mkldnn=self.enable_mkldnn,
                    cpu_threads=self.cpu_threads,
                    mkldnn_cache_capacity=self.mkldnn_cache_capacity,
                )
                _MODEL_CACHE[det_key] = self.text_detector
            except Exception as e:
                logger.warning(f"FullPageTextDetector init failed: {e}")
                self.text_detector = None

        # Two-Source Region Reconstructor
        self.reconstructor = RegionReconstructor(config=self.config)
        self.profiler = None

    def set_profiler(self, profiler: Any) -> None:
        """Attach a PipelineProfiler to instrument the detector and its subcomponents."""
        self.profiler = profiler
        if hasattr(self, 'refiner') and self.refiner is not None:
            self.refiner.profiler = profiler
            if hasattr(self.refiner, 'line_analyzer') and self.refiner.line_analyzer is not None:
                self.refiner.line_analyzer.profiler = profiler
        if hasattr(self, 'text_detector') and self.text_detector is not None:
            self.text_detector.profiler = profiler
        if hasattr(self, 'reconstructor') and self.reconstructor is not None:
            self.reconstructor.profiler = profiler

    # Main entry point

    def detect(self, image_path: str,
               original_image: np.ndarray,
               confidence: float = 0.15,
               profiler: Any = None) -> List[Dict[str, Any]]:
        """
        Run the full detection pipeline.

        Args:
            image_path: Path to the original image (for PP-DocLayoutV2).
            original_image: Original BGR image array.
            confidence: Minimum confidence threshold.

        Returns:
            List of detection dicts [{bbox, class, confidence}, ...]
        """
        h, w = original_image.shape[:2]
        if profiler is not None:
            self.set_profiler(profiler)
        prof = getattr(self, 'profiler', None)

        # Preprocessing: Manuscript boundary isolation
        if prof:
            with prof.timer('preprocessing', pixels=h * w):
                ms_bbox = self._find_manuscript_region(original_image)
        else:
            ms_bbox = self._find_manuscript_region(original_image)

        # Source 1: PP-DocLayoutV2 Macro Layout Prior
        macro_proposals: List[Dict[str, Any]] = []
        if self.model_available:
            if prof:
                with prof.timer('layout_inference', pixels=h * w):
                    raw = self._run_model(image_path)
                prof.record_count('pp_doclayoutv2_calls', 1 if raw is not None else 0)
            else:
                raw = self._run_model(image_path)

            if raw:
                if prof:
                    with prof.timer('heuristics', pixels=h * w, is_substage_context=True):
                        mapped = self._map_classes(raw, w, h, ms_bbox, original_image)
                else:
                    mapped = self._map_classes(raw, w, h, ms_bbox, original_image)
                macro_proposals.extend(mapped)
                logger.debug(
                    f"PP-DocLayoutV2: {len(raw)} raw -> "
                    f"{len(mapped)} mapped detections"
                )

        # Source 2: Full-Page PP-OCRv5 Fine-Grained Text Detection
        text_boxes: List[Dict[str, Any]] = []
        if hasattr(self, 'text_detector') and self.text_detector and self.text_detector.is_available:
            if prof:
                with prof.timer('text_detection'):
                    text_boxes = self.text_detector.detect(original_image)
            else:
                text_boxes = self.text_detector.detect(original_image)
            logger.debug(f"[TWO-SOURCE] PP-OCRv5 full-page text detector: {len(text_boxes)} text boxes")
        self._last_text_boxes = text_boxes

        # Full-page line ownership is the primary classifier.  The layout model
        # supplies priors and the reconstructor supplies normalized structure;
        # neither source is allowed to finalize a broad semantic box.
        ownership_pipeline = False
        if text_boxes and hasattr(self, 'reconstructor') and self.reconstructor:
            structure = self.reconstructor.analyze_structure(original_image, text_boxes, macro_proposals)
            if prof:
                with prof.timer('region_reconstruction', pixels=h * w):
                    detections = self.refiner.refine_page(original_image, text_boxes, structure, macro_proposals)
            else:
                detections = self.refiner.refine_page(original_image, text_boxes, structure, macro_proposals)

            # Fine-grained text ownership is authoritative for text semantics.
            # Preserve only genuinely visual macro objects (image/seal/chart/
            # formula-like regions) that do not overlap owned text. Broad macro
            # header/footer/main/side proposals are intentionally not copied.
            visual_fillers = self._collect_visual_macro_fillers(original_image, macro_proposals, detections)
            if visual_fillers:
                detections.extend(visual_fillers)
                logger.debug(f"[TWO-SOURCE] Added {len(visual_fillers)} visual filler proposals")

            ownership_pipeline = True
            logger.debug(f"[TWO-SOURCE] Rebuilt {len(detections)} layout regions from line ownership")
        else:
            detections = list(macro_proposals)

        # Compatibility path for runs without fine-grained OCR components.
        if detections and not ownership_pipeline:
            if prof:
                with prof.timer('semantic_refinement', is_substage_context=True):
                    detections = self.refiner.refine_detections(original_image, detections)
            else:
                detections = self.refiner.refine_detections(original_image, detections)

        # Strip scans are retained only as a compatibility fallback.  On the
        # line-owned path they would reintroduce the very broad strip boxes the
        # ownership stage has removed.
        if not ownership_pipeline:
            if prof:
                with prof.timer('filler_detection', is_substage_context=True):
                    filler = self._detect_filler(original_image, ms_bbox, existing_detections=detections)
            else:
                filler = self._detect_filler(original_image, ms_bbox, existing_detections=detections)
            if filler:
                detections.extend(filler)
                logger.debug(f"Filler refinement: {len(filler)} regions")

            if prof:
                with prof.timer('heuristics', is_substage_context=True):
                    footer = self._detect_footer(original_image, ms_bbox, existing_detections=detections)
            else:
                footer = self._detect_footer(original_image, ms_bbox, existing_detections=detections)
            if footer:
                detections.extend(footer)
                logger.debug(f"Footer detection: {len(footer)} regions")
        return detections

    # Layer 1: PP-DocLayoutV2 inference

    def _run_model(self, image_path: str) -> List[Dict[str, Any]]:
        """Run PP-DocLayoutV2 and return raw detections."""
        raw: List[Dict[str, Any]] = []
        try:
            abs_image_path = os.path.abspath(image_path)
            with suppress_c_output():
                outputs = self.model.predict(
                    abs_image_path, batch_size=1, layout_nms=True,
                )

            for result in outputs:
                boxes_info = result.get('boxes', [])
                if not boxes_info:
                    # Try alternate output format
                    if 'result' in result:
                        boxes_info = result['result']

                if isinstance(boxes_info, list):
                    for box in boxes_info:
                        det = self._parse_model_output(box)
                        if det:
                            raw.append(det)

        except Exception as e:
            logger.warning(f"PP-DocLayoutV2 inference error: {e}")

        return raw

    def _parse_model_output(self, box: Any) -> Optional[Dict[str, Any]]:
        """Parse a single PP-DocLayoutV2 output box into our format."""
        try:
            # PaddleOCR LayoutDetection output format
            if isinstance(box, dict):
                coords = box.get('coordinate', box.get('bbox', []))
                label = str(box.get('label', box.get('cls_name', 'unknown'))).lower().strip()
                score = float(box.get('score', box.get('confidence', 0.0)))

                if len(coords) >= 4:
                    return {
                        'bbox': {
                            'x_min': int(coords[0]),
                            'y_min': int(coords[1]),
                            'x_max': int(coords[2]),
                            'y_max': int(coords[3]),
                        },
                        'raw_label': label,
                        'confidence': score,
                    }
        except Exception as e:
            logger.debug(f"Failed to parse box: {box} — {e}")

        return None

    # Layer 2: Class mapping & Structural Grid Decomposition

    def _decompose_ruled_container(self, image: np.ndarray,
                                    bbox: Dict[str, int],
                                    ms_bbox: Dict[str, int],
                                    conf: float = 0.90) -> List[Dict[str, Any]]:
        """
        Decomposes a multi-column framed/ruled composite container into:
        - header: top section / title row
        - side_text: left margin commentary column
        - main_text: central Sanskrit root verses
        - side_text: right margin commentary column
        - footer: bottom page / folio marker
        
        Returns empty list if container lacks internal vertical column dividers (N_v < 2).
        """
        h, w = image.shape[:2]
        x1 = max(0, int(bbox['x_min']))
        y1 = max(0, int(bbox['y_min']))
        x2 = min(w, int(bbox['x_max']))
        y2 = min(h, int(bbox['y_max']))

        bw = x2 - x1
        bh = y2 - y1
        if bw < 100 or bh < 100:
            return []

        roi = image[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]

        # Extract red and dark framing rules in ROI
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 35, 35]), np.array([14, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([166, 35, 35]), np.array([180, 255, 255]))
        mask_lines = m1 | m2

        # If red lines are sparse, also check dark inked rules
        if np.sum(mask_lines) < rw * rh * 0.005:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 25, 10)
            mask_lines = mask_lines | adapt

        # Morphological horizontal and vertical rule extraction
        h_len = max(15, int(rw * 0.15))
        v_len = max(15, int(rh * 0.15))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

        h_lines = cv2.morphologyEx(mask_lines, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(mask_lines, cv2.MORPH_OPEN, v_kernel)

        h_proj = np.sum(h_lines, axis=1)
        v_proj = np.sum(v_lines, axis=0)

        h_peaks = np.where(h_proj > rw * 0.08 * 255)[0]
        v_peaks = np.where(v_proj > rh * 0.08 * 255)[0]

        def _group_peaks(peaks, min_gap=25):
            if len(peaks) == 0:
                return []
            groups = []
            cur = [peaks[0]]
            for p in peaks[1:]:
                if p - cur[-1] <= min_gap:
                    cur.append(p)
                else:
                    groups.append(int(np.mean(cur)))
                    cur = [p]
            groups.append(int(np.mean(cur)))
            return groups

        h_divs = _group_peaks(h_peaks, min_gap=25)
        v_divs = _group_peaks(v_peaks, min_gap=25)

        # Discard bounding frame lines near borders
        int_h = [y for y in h_divs if 0.04 * rh < y < 0.96 * rh]
        int_v = [x for x in v_divs if 0.04 * rw < x < 0.96 * rw]

        # Multi-column grid requires at least 2 vertical divider lines (flanking central text)
        if len(int_v) < 2:
            return []

        from src.postprocessor import tighten_bbox_to_content

        v0 = int_v[0]
        v1 = int_v[-1]

        # Central column must be significant in width
        if (v1 - v0) < 0.20 * rw or (v1 - v0) < 120:
            return []

        # Determine horizontal divisions: header at top, footer at bottom ONLY if physical dividing lines exist
        h0 = int_h[0] if len(int_h) >= 1 else 0
        h1 = int_h[-1] if len(int_h) >= 2 else rh

        sub_regions = []

        # 1. Header (top row above h0 only if horizontal divider exists)
        if h0 > 15:
            hdr_box = {'x_min': x1, 'y_min': y1, 'x_max': x2, 'y_max': y1 + h0}
            tight_hdr, dens = tighten_bbox_to_content(image, hdr_box)
            if (tight_hdr['x_max'] - tight_hdr['x_min'] > 25) and (tight_hdr['y_max'] - tight_hdr['y_min'] > 10) and dens >= 0.03:
                sub_regions.append({
                    'bbox': tight_hdr,
                    'class': 'header',
                    'confidence': round(conf, 4),
                    'layout_conf': round(conf, 4),
                })

        # 2. Middle Row: from h0 to h1
        mid_y0 = y1 + h0
        mid_y1 = y1 + h1

        # Left Column: side_text (commentary)
        left_box = {'x_min': x1, 'y_min': mid_y0, 'x_max': x1 + v0, 'y_max': mid_y1}
        tight_left, dens = tighten_bbox_to_content(image, left_box)
        if (tight_left['x_max'] - tight_left['x_min'] > 20) and (tight_left['y_max'] - tight_left['y_min'] > 15) and dens >= 0.03:
            sub_regions.append({
                'bbox': tight_left,
                'class': 'side_text',
                'confidence': round(conf, 4),
                'layout_conf': round(conf, 4),
            })

        # Center Column: main_text (root verses)
        center_box = {'x_min': x1 + v0, 'y_min': mid_y0, 'x_max': x1 + v1, 'y_max': mid_y1}
        tight_center, dens = tighten_bbox_to_content(image, center_box)
        if (tight_center['x_max'] - tight_center['x_min'] > 25) and (tight_center['y_max'] - tight_center['y_min'] > 15) and dens >= 0.03:
            sub_regions.append({
                'bbox': tight_center,
                'class': 'main_text',
                'confidence': round(conf, 4),
                'layout_conf': round(conf, 4),
            })

        # Right Column: side_text (commentary)
        right_box = {'x_min': x1 + v1, 'y_min': mid_y0, 'x_max': x2, 'y_max': mid_y1}
        tight_right, dens = tighten_bbox_to_content(image, right_box)
        if (tight_right['x_max'] - tight_right['x_min'] > 20) and (tight_right['y_max'] - tight_right['y_min'] > 15) and dens >= 0.03:
            sub_regions.append({
                'bbox': tight_right,
                'class': 'side_text',
                'confidence': round(conf, 4),
                'layout_conf': round(conf, 4),
            })

        # 3. Footer (bottom row below h1 only if bottom horizontal divider exists)
        if h1 < rh and (rh - h1) > 15:
            ftr_box = {'x_min': x1, 'y_min': y1 + h1, 'x_max': x2, 'y_max': y2}
            tight_ftr, dens = tighten_bbox_to_content(image, ftr_box)
            if (tight_ftr['x_max'] - tight_ftr['x_min'] > 15) and (tight_ftr['y_max'] - tight_ftr['y_min'] > 10) and dens >= 0.03:
                sub_regions.append({
                    'bbox': tight_ftr,
                    'class': 'footer',
                    'confidence': round(conf, 4),
                    'layout_conf': round(conf, 4),
                })

        return sub_regions

    def _collect_visual_macro_fillers(self, image: np.ndarray,
                                      macro_proposals: List[Dict[str, Any]],
                                      owned_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only macro detections that represent real non-text visual objects.

        This is a conservative fusion rule: a coarse ``filler`` prediction is
        never allowed to swallow OCR-owned manuscript text.
        """
        visual_labels = {'image', 'seal', 'chart', 'formula_number', 'inline_formula', 'formula'}
        h, w = image.shape[:2]
        text_regions = [r for r in owned_regions if r.get('class') in {'header', 'main_text', 'footer', 'side_text'}]
        output = []
        for proposal in macro_proposals:
            if proposal.get('class') != 'filler' or proposal.get('macro_source') not in visual_labels:
                continue
            b = proposal.get('bbox', {})
            x1, y1 = max(0, int(b.get('x_min', 0))), max(0, int(b.get('y_min', 0)))
            x2, y2 = min(w, int(b.get('x_max', 0))), min(h, int(b.get('y_max', 0)))
            if x2 <= x1 or y2 <= y1:
                continue
            area = max(1, (x2 - x1) * (y2 - y1))
            if area > 0.40 * w * h:
                continue

            heavy_overlap = False
            for region in text_regions:
                rb = region.get('bbox', {})
                ix1, iy1 = max(x1, rb.get('x_min', 0)), max(y1, rb.get('y_min', 0))
                ix2, iy2 = min(x2, rb.get('x_max', 0)), min(y2, rb.get('y_max', 0))
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter / area >= 0.30:
                    heavy_overlap = True
                    break
            if heavy_overlap:
                continue

            from src.postprocessor import tighten_bbox_to_content
            tight, density = tighten_bbox_to_content(image, {
                'x_min': x1, 'y_min': y1, 'x_max': x2, 'y_max': y2
            }, pad=2)
            if tight['x_max'] - tight['x_min'] < 8 or tight['y_max'] - tight['y_min'] < 8:
                continue
            if density < 0.005:
                continue
            output.append({
                'bbox': tight,
                'class': 'filler',
                'confidence': round(float(proposal.get('confidence', 0.80)), 4),
                'layout_conf': round(float(proposal.get('confidence', 0.80)), 4),
                'geometry_source': 'macro_visual',
                'macro_source': proposal.get('macro_source'),
            })
        return output

    def _map_classes(self, raw: List[Dict[str, Any]],
                      img_w: int, img_h: int,
                      ms_bbox: Dict[str, int],
                      image: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Map PP-DocLayoutV2 labels to assignment classes:
        header, footer, main_text, side_text, filler.
        Uses direct_map, candidate_map, and structural validation with abstention.
        """
        mapped: List[Dict[str, Any]] = []

        for det in raw:
            label = det['raw_label']
            bbox = det['bbox']

            if label in self.ignore_labels:
                continue

            # Check if this detection is a composite container (or large table/content)
            cand_target = self.candidate_map.get(label, '')
            is_composite = (cand_target == 'composite_container' or label in ['table', 'content'])
            bw = bbox['x_max'] - bbox['x_min']
            bh = bbox['y_max'] - bbox['y_min']
            ms_w = max(1, ms_bbox['x_max'] - ms_bbox['x_min'])
            ms_h = max(1, ms_bbox['y_max'] - ms_bbox['y_min'])
            box_area = bw * bh
            ms_area = ms_w * ms_h

            if image is not None and is_composite:
                decomposed = self._decompose_ruled_container(image, bbox, ms_bbox, det.get('confidence', 0.90))
                if decomposed:
                    logger.debug(f"[GRID] Decomposed container {bbox} into {len(decomposed)} sub-regions")
                    mapped.extend(decomposed)
                    continue

            # Direct mapping - preserve model's original semantics (no naive flipping to header/footer)
            if label in self.direct_map:
                cls = self.direct_map[label]
            elif label in self.candidate_map:
                cls = self._validate_candidate(
                    det, self.candidate_map[label],
                    img_w, img_h, ms_bbox,
                )
            else:
                cls = self._spatial_classify(bbox, img_w, img_h, ms_bbox)

            if cls is None:
                continue

            mapped.append({
                'bbox': bbox,
                'class': cls,
                'confidence': det['confidence'],
                'macro_source': label,
                'geometry_source': 'macro_prior',
            })

        # Merge vertically adjacent side_text segments in the same margin column
        side_dets = [d for d in mapped if d['class'] == 'side_text']
        other_dets = [d for d in mapped if d['class'] != 'side_text']
        if len(side_dets) > 1:
            merged_side = []
            used = set()
            for i in range(len(side_dets)):
                if i in used:
                    continue
                cur = dict(side_dets[i])
                cur_b = dict(cur['bbox'])
                for j in range(i + 1, len(side_dets)):
                    if j in used:
                        continue
                    b2 = side_dets[j]['bbox']
                    x_overlap = min(cur_b['x_max'], b2['x_max']) - max(cur_b['x_min'], b2['x_min'])
                    min_w = min(cur_b['x_max'] - cur_b['x_min'], b2['x_max'] - b2['x_min'])
                    y_gap = max(0, max(cur_b['y_min'], b2['y_min']) - min(cur_b['y_max'], b2['y_max']))
                    if (x_overlap / max(1, min_w) > 0.40) and y_gap <= 35:
                        cur_b['x_min'] = min(cur_b['x_min'], b2['x_min'])
                        cur_b['y_min'] = min(cur_b['y_min'], b2['y_min'])
                        cur_b['x_max'] = max(cur_b['x_max'], b2['x_max'])
                        cur_b['y_max'] = max(cur_b['y_max'], b2['y_max'])
                        cur['confidence'] = max(cur['confidence'], side_dets[j]['confidence'])
                        used.add(j)
                cur['bbox'] = cur_b
                merged_side.append(cur)
            mapped = other_dets + merged_side

        return mapped

    def _validate_candidate(self, det: Dict, candidate_cls: str,
                              img_w: int, img_h: int,
                              ms_bbox: Dict[str, int]) -> Optional[str]:
        """
        Validate a candidate mapping using strict structural and evidence gates.
        Candidate mappings MUST NOT be treated as unconditional.
        Returns one of the 5 assignment classes or None to discard (abstention).
        """
        raw_label = det.get('raw_label', '')
        conf = det.get('confidence', 1.0)
        bbox = det['bbox']
        bw = bbox['x_max'] - bbox['x_min']
        bh = bbox['y_max'] - bbox['y_min']
        box_area = bw * bh

        ms_w = max(1, ms_bbox['x_max'] - ms_bbox['x_min'])
        ms_h = max(1, ms_bbox['y_max'] - ms_bbox['y_min'])
        ms_area = ms_w * ms_h

        y_center = (bbox['y_min'] + bbox['y_max']) / 2
        x_center = (bbox['x_min'] + bbox['x_max']) / 2
        y_rel = (y_center - ms_bbox['y_min']) / ms_h
        x_rel = (x_center - ms_bbox['x_min']) / ms_w
        w_ratio = bw / ms_w

        # 1. Marginalia candidates (aside_text, reference, reference_content, footnote)
        if candidate_cls == 'side_text':
            ar = bh / max(1, bw)
            if bw < 30 or ar > 8.0 or (conf < 0.25 and bw < 60):
                return None
            if x_rel < self.side_margin or x_rel > (1 - self.side_margin):
                return 'side_text'
            # Bottom attribution in extreme bottom margin is filler
            if raw_label in {'footnote', 'vision_footnote'} and y_rel >= 0.88:
                return 'filler'
            if raw_label in {'footnote', 'vision_footnote'} and y_rel > self.footer_zone:
                return 'footer'
            return 'main_text'

        # 2. Header candidates (paragraph_title, number)
        if candidate_cls == 'header':
            if y_rel < self.header_zone and w_ratio < 0.85:
                return 'header'
            if raw_label == 'number':
                if y_rel > self.footer_zone:
                    return 'footer'
                if x_rel < self.side_margin or x_rel > (1 - self.side_margin):
                    return 'side_text'
                return 'main_text'
            return 'main_text'

        # 3. Filler candidates (image, seal, chart, formula_number, inline_formula)
        if candidate_cls == 'filler':
            if box_area >= 0.50 * ms_area:
                logger.debug(f"Rejected oversized filler candidate covering >50% page: {bbox}")
                return None
            if bw < 15 or bh < 15 or (box_area < 0.0002 * ms_area):
                return None
            return 'filler'

        # 4. Main text candidates (table, vertical_text)
        if candidate_cls == 'main_text':
            if (x_rel < self.side_margin or x_rel > (1 - self.side_margin)) and w_ratio < 0.28:
                return 'side_text'
            return 'main_text'

        return candidate_cls

    def _spatial_classify(self, bbox: Dict[str, int],
                           img_w: int, img_h: int,
                           ms_bbox: Dict[str, int]) -> Optional[str]:
        """Classify an unmapped detection by its position within the manuscript."""
        ms_h = max(1, ms_bbox['y_max'] - ms_bbox['y_min'])
        ms_w = max(1, ms_bbox['x_max'] - ms_bbox['x_min'])

        y_center = (bbox['y_min'] + bbox['y_max']) / 2
        x_center = (bbox['x_min'] + bbox['x_max']) / 2

        y_rel = (y_center - ms_bbox['y_min']) / ms_h
        x_rel = (x_center - ms_bbox['x_min']) / ms_w

        if y_rel < self.header_zone:
            return 'header'
        if y_rel > self.footer_zone:
            return 'footer'
        if x_rel < self.side_margin or x_rel > (1 - self.side_margin):
            bw = bbox['x_max'] - bbox['x_min']
            bh = bbox['y_max'] - bbox['y_min']
            ar = bh / max(1, bw)
            if bw < 50 or ar > 8.0:
                return None
            return 'side_text'
        return 'main_text'

    # Second-Stage Content Analysis delegation

    def _inspect_and_split_main_text(self, image: np.ndarray,
                                     main_det: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Delegate to SemanticRefiner."""
        if not hasattr(self, 'refiner') or self.refiner is None:
            from src.semantic_refiner import SemanticRefiner
            self.refiner = SemanticRefiner(recognizer=self.recognizer, config=self.config)
        return self.refiner._inspect_and_split_main_text(image, main_det)

    # Manuscript region isolation

    def _find_manuscript_region(self, image: np.ndarray) -> Dict[str, int]:
        """
        Isolate the manuscript page from background using
        Otsu colour segmentation + horizontal projection refinement.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > w * h * 0.10:
                bx, by, bw, bh = cv2.boundingRect(largest)
                roi = gray[by:by + bh, bx:bx + bw]
                return self._refine_bounds(roi, bx, by, bw, bh, w, h)

        # Horizontal projection fallback
        brightness = np.mean(gray, axis=1)
        thr = np.mean(brightness) + 0.1 * np.std(brightness)
        bright = np.where(brightness > thr)[0]
        if len(bright) > 10:
            ys, ye = int(bright[0]), int(bright[-1])
            if ye - ys > h * 0.15:
                return {'x_min': 0, 'y_min': ys, 'x_max': w, 'y_max': ye}

        mx, my = int(w * 0.05), int(h * 0.12)
        return {'x_min': mx, 'y_min': my, 'x_max': w - mx, 'y_max': h - my}

    def _refine_bounds(self, roi: np.ndarray,
                        bx: int, by: int, bw: int, bh: int,
                        img_w: int, img_h: int) -> Dict[str, int]:
        """Tighten manuscript bounds via horizontal text projection."""
        binary = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 8,
        )
        h_proj = np.sum(binary, axis=1) / 255.0
        text_rows = np.where(h_proj > bw * 0.05)[0]

        if len(text_rows) > 5:
            pad = max(5, bh // 40)
            ys = max(0, int(text_rows[0]) - pad)
            ye = min(bh, int(text_rows[-1]) + pad)
        else:
            ys, ye = 0, bh

        return {
            'x_min': max(0, bx),
            'y_min': max(0, by + ys),
            'x_max': min(img_w, bx + bw),
            'y_max': min(img_h, by + ye),
        }

    # Layer 3: Filler detection (English text, decorative, pencil)

    def _detect_filler(self, image: np.ndarray,
                        ms_bbox: Dict[str, int],
                        existing_detections: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Detect filler regions: English attribution text above/below
        the manuscript, decorative elements, pencil text.
        Tightens to actual text content, avoiding photographic background.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dets: List[Dict[str, Any]] = []

        ms_top, ms_bot = ms_bbox['y_min'], ms_bbox['y_max']

        # Check if the pretrained detector already captured top or bottom attribution
        has_top_filler = any(
            d.get('class') == 'filler' and d['bbox']['y_max'] < 0.20 * h
            for d in (existing_detections or [])
        )
        has_bot_filler = any(
            d.get('class') == 'filler' and d['bbox']['y_min'] > 0.80 * h
            for d in (existing_detections or [])
        )

        # Attribution text ABOVE manuscript (only if not already detected by model)
        if not has_top_filler and ms_top > h * 0.03:
            box = self._find_text_in_strip(gray[0:ms_top, :], 0, 0, w, ms_top)
            if box:
                line_info = self.refiner.line_analyzer._analyze_single_line_crop(
                    image, box['x_min'], box['y_min'], box['x_max'], box['y_max'], 0, 0, box['y_max'] - box['y_min']
                )
                if line_info:
                    sc = self.refiner.calculate_filler_score(line_info)
                    if sc >= self.refiner.score_threshold:
                        f_conf = round(float(np.clip(0.60 * sc + 0.40 * line_info.get('ocr_conf', 0.80), 0.10, 0.99)), 4)
                        dets.append({
                            'bbox': line_info['bbox'],
                            'class': 'filler',
                            'confidence': f_conf,
                            'ocr_conf': line_info.get('ocr_conf', 0.0),
                            'deva_ratio': line_info.get('deva_ratio', 0.0),
                            'latin_ratio': line_info.get('latin_ratio', 0.0),
                        })

        # Attribution text BELOW manuscript (only if not already detected by model)
        if not has_bot_filler and ms_bot < h * 0.97:
            box = self._find_text_in_strip(
                gray[ms_bot:h, :], 0, ms_bot, w, h,
            )
            if box:
                line_info = self.refiner.line_analyzer._analyze_single_line_crop(
                    image, box['x_min'], box['y_min'], box['x_max'], box['y_max'], 0, 0, box['y_max'] - box['y_min']
                )
                if line_info:
                    sc = self.refiner.calculate_filler_score(line_info)
                    if sc >= self.refiner.score_threshold:
                        f_conf = round(float(np.clip(0.60 * sc + 0.40 * line_info.get('ocr_conf', 0.80), 0.10, 0.99)), 4)
                        dets.append({
                            'bbox': line_info['bbox'],
                            'class': 'filler',
                            'confidence': f_conf,
                            'ocr_conf': line_info.get('ocr_conf', 0.0),
                            'deva_ratio': line_info.get('deva_ratio', 0.0),
                            'latin_ratio': line_info.get('latin_ratio', 0.0),
                        })

        # Printed overlay inside top edge of manuscript (tight text search)
        if not has_top_filler:
            ms_h = ms_bot - ms_top
            strip_h = max(10, int(ms_h * 0.08))
            top_strip = gray[ms_top:ms_top + strip_h, :]
            box = self._find_text_in_strip(top_strip, ms_bbox['x_min'], ms_top, ms_bbox['x_max'], ms_top + strip_h)
            if box:
                line_info = self.refiner.line_analyzer._analyze_single_line_crop(
                    image, box['x_min'], box['y_min'], box['x_max'], box['y_max'], 0, 0, box['y_max'] - box['y_min']
                )
                if line_info:
                    sc = self.refiner.calculate_filler_score(line_info)
                    if sc >= self.refiner.score_threshold:
                        f_conf = round(float(np.clip(0.60 * sc + 0.40 * line_info.get('ocr_conf', 0.80), 0.10, 0.99)), 4)
                        dets.append({
                            'bbox': line_info['bbox'],
                            'class': 'filler',
                            'confidence': f_conf,
                            'ocr_conf': line_info.get('ocr_conf', 0.0),
                            'deva_ratio': line_info.get('deva_ratio', 0.0),
                            'latin_ratio': line_info.get('latin_ratio', 0.0),
                        })

        return dets

    def _find_text_in_strip(self, roi: np.ndarray,
                              ox: int, oy: int,
                              mx: int, my: int) -> Optional[Dict[str, int]]:
        """
        Detects actual text regions inside a margin strip using character clustering.
        Tightens directly to the actual text glyphs, rejecting noise or photographic background.
        """
        sh, sw = roi.shape[:2]
        if sh < 6 or sw < 20:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        mean_val = np.mean(gray)

        if mean_val < 100:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, -4
            )
        else:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, 4
            )

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        char_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if 4 <= h <= sh * 0.95 and 2 <= w <= sw * 0.35 and area >= 4:
                char_boxes.append((x, y, w, h))

        if len(char_boxes) < 6:
            return None

        char_boxes.sort(key=lambda b: b[0])
        clusters = []
        cur_cluster = [char_boxes[0]]

        for b in char_boxes[1:]:
            prev_b = cur_cluster[-1]
            gap = b[0] - (prev_b[0] + prev_b[2])
            if gap <= 30 and abs(b[1] - prev_b[1]) <= 10:
                cur_cluster.append(b)
            else:
                if len(cur_cluster) >= 6:
                    clusters.append(cur_cluster)
                cur_cluster = [b]
        if len(cur_cluster) >= 6:
            clusters.append(cur_cluster)

        if not clusters:
            return None

        best = max(clusters, key=len)
        x1 = min(b[0] for b in best)
        y1 = min(b[1] for b in best)
        x2 = max(b[0] + b[2] for b in best)
        y2 = max(b[1] + b[3] for b in best)

        char_area = sum(b[2] * b[3] for b in best)
        box_area = max(1, (x2 - x1) * (y2 - y1))
        density = char_area / box_area

        if (x2 - x1) >= 25 and density >= 0.08:
            pad = 2
            return {
                'x_min': max(0, ox + x1 - pad),
                'y_min': max(0, oy + y1 - pad),
                'x_max': min(mx, ox + x2 + pad),
                'y_max': min(my, oy + y2 + pad),
            }
        return None

    # Layer 3b: Footer detection

    def _detect_footer(self, image: np.ndarray,
                       ms_bbox: Dict[str, int],
                       existing_detections: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Detect footer regions (Devanagari page numbers, catchwords,
        colophon markers) below the main manuscript body.

        Only adds a footer when:
        - No footer already exists in detections
        - Text is found below the main text body with a clear gap
        - The text is NOT English attribution (verified via filler scoring)
        """
        h, w = image.shape[:2]
        dets: List[Dict[str, Any]] = []

        main_texts = [
            d for d in (existing_detections or [])
            if d.get('class') == 'main_text'
        ]
        if not main_texts:
            return dets

        main_bottom = max(d['bbox']['y_max'] for d in main_texts)
        ms_bot = ms_bbox['y_max']

        # Scan from main_text bottom to manuscript bottom + small margin
        scan_top = main_bottom
        scan_bot = min(h, ms_bot + int(h * 0.06))
        if scan_bot - scan_top < 15:
            # Main text already extends to bottom of manuscript region; no gap for footer
            return dets

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        strip = gray[scan_top:scan_bot, ms_bbox['x_min']:ms_bbox['x_max']]
        sh, sw = strip.shape[:2]
        if sh < 8 or sw < 20:
            return dets

        # Adaptive binarisation
        mean_val = np.mean(strip)
        if mean_val < 100:
            binary = cv2.adaptiveThreshold(
                strip, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, -4,
            )
        else:
            binary = cv2.adaptiveThreshold(
                strip, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, 4,
            )

        h_proj = np.sum(binary, axis=1) / 255.0
        text_thr = max(1.0, sw * 0.01)
        text_rows = np.where(h_proj >= text_thr)[0]

        if len(text_rows) < 3:
            return dets

        # Require a clear vertical gap between main text bottom and footer content
        first_row = int(text_rows[0])
        if first_row < 20:
            return dets

        last_row = int(text_rows[-1])
        footer_height = last_row - first_row
        if footer_height > 60:
            return dets

        text_part = binary[first_row:last_row + 1, :]
        col_proj = np.sum(text_part, axis=0)
        text_cols = np.where(col_proj > 0)[0]
        if len(text_cols) < 3:
            return dets

        pad = 3
        fx1 = max(0, ms_bbox['x_min'] + int(text_cols[0]) - pad)
        fy1 = max(0, scan_top + first_row - pad)
        fx2 = min(w, ms_bbox['x_min'] + int(text_cols[-1]) + pad)
        fy2 = min(h, scan_top + last_row + 1 + pad)

        if (fx2 - fx1) < 15 or (fy2 - fy1) < 6:
            return dets
        if (fx2 - fx1) > 0.75 * (ms_bbox['x_max'] - ms_bbox['x_min']):
            return dets

        cand_bbox = {'x_min': fx1, 'y_min': fy1, 'x_max': fx2, 'y_max': fy2}
        # Do not add duplicate footer if it substantially overlaps an existing footer
        if any(_iou(cand_bbox, d['bbox']) > 0.35 for d in (existing_detections or []) if d.get('class') == 'footer'):
            return dets

        # Verify this is valid text and NOT English attribution (filler)
        line_info = None
        if hasattr(self, 'refiner') and self.refiner is not None and hasattr(self.refiner, 'line_analyzer'):
            try:
                line_info = self.refiner.line_analyzer._analyze_single_line_crop(
                    image, fx1, fy1, fx2, fy2, 0, 0, fy2 - fy1
                )
            except Exception:
                line_info = None

        ocr_conf = 0.80
        if line_info:
            ocr_txt = line_info.get('ocr_text', '').strip()
            ocr_conf = line_info.get('ocr_conf', 0.80)
            filler_score = self.refiner.calculate_filler_score(line_info)
            if filler_score >= self.refiner.score_threshold:
                return dets  # English attribution — handled as filler

            # Conservative fallback: a footer must have positive manuscript-like
            # OCR evidence.  Pixel ink alone is not enough, otherwise the final
            # body line of a damaged scan can be promoted to footer.
            deva_ratio = float(line_info.get('deva_ratio', 0.0) or 0.0)
            digit_text = bool(re.search(r'\d', ocr_txt))
            if not ocr_txt or (deva_ratio < 0.20 and not digit_text):
                return dets
        else:
            # No recognizer evidence -> abstain rather than hallucinating a
            # footer from dark pixels.
            return dets

        footer_conf = round(float(np.clip(ocr_conf, 0.10, 0.95)), 4)
        dets.append({
            'bbox': {'x_min': fx1, 'y_min': fy1, 'x_max': fx2, 'y_max': fy2},
            'class': 'footer',
            'confidence': footer_conf,
        })
        logger.debug(f"[FOOTER] Detected footer: x=[{fx1},{fx2}], y=[{fy1},{fy2}], conf={footer_conf}")
        return dets
