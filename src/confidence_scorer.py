"""
Evidence-Based Normalized Confidence Scorer.

Combines multiple independent signals into a principled, normalized confidence score:
1. layout: PP-DocLayoutV2 prior confidence
2. ocr_det: Fine-grained DBNet text detection confidence
3. ocr_rec: OCR character/token recognition confidence
4. geometry: Foreground ink density and aspect ratio regularity
5. semantic: Script consistency (Devanagari vs Latin alignment with class definition)

Zero fabrication: No arbitrary minimum clamping (e.g. no max(0.88, score)).
All weights are configurable, documented, and dynamically normalized when signals are missing.
"""

import logging
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger('manuscript-detector')


class ConfidenceScorer:
    """
    Computes an evidence-based normalized confidence score for manuscript regions.
    """

    DEFAULT_WEIGHTS = {
        'layout': 0.25,     # Macro layout prior (PP-DocLayoutV2)
        'ocr_det': 0.25,    # Fine-grained text box detection (DBNet)
        'ocr_rec': 0.20,    # Text recognition confidence (PP-OCRv5)
        'geometry': 0.15,   # Ink density and aspect ratio regularity
        'semantic': 0.15,   # Script alignment with assignment class definition
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        cs_cfg = self.config.get('confidence_scoring', {})
        self.weights = cs_cfg.get('weights', self.DEFAULT_WEIGHTS)

    def compute_confidence(
        self,
        det: Dict[str, Any],
        image: Optional[np.ndarray] = None,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> float:
        """
        Compute normalized confidence for a detection based on available evidence signals.
        Returns float in [0.01, 0.99].
        """
        signals: Dict[str, float] = {}

        # Signal 1: Macro Layout Detection Confidence
        if 'layout_conf' in det and det['layout_conf'] is not None:
            signals['layout'] = float(np.clip(det['layout_conf'], 0.0, 1.0))
        elif 'raw_confidence' in det and det['raw_confidence'] is not None:
            signals['layout'] = float(np.clip(det['raw_confidence'], 0.0, 1.0))

        # Signal 2: OCR Text Detection Confidence (DBNet)
        if 'ocr_det_conf' in det and det['ocr_det_conf'] is not None:
            signals['ocr_det'] = float(np.clip(det['ocr_det_conf'], 0.0, 1.0))
        elif 'line_scores' in det and det['line_scores']:
            signals['ocr_det'] = float(np.clip(np.mean(det['line_scores']), 0.0, 1.0))

        # Signal 3: OCR Text Recognition Confidence
        if 'ocr_conf' in det and det['ocr_conf'] is not None and float(det['ocr_conf']) > 0.0:
            signals['ocr_rec'] = float(np.clip(det['ocr_conf'], 0.0, 1.0))
        elif 'ocr_confidence' in det and det['ocr_confidence'] is not None and float(det['ocr_confidence']) > 0.0:
            signals['ocr_rec'] = float(np.clip(det['ocr_confidence'], 0.0, 1.0))

        # Signal 4: Geometry Confidence (Ink content density & aspect ratio)
        geom_score = self._compute_geometry_score(det, image, image_shape)
        if geom_score is not None:
            signals['geometry'] = float(np.clip(geom_score, 0.0, 1.0))

        # Signal 5: Semantic Confidence (Script ratio per class definition)
        sem_score = self._compute_semantic_score(det)
        if sem_score is not None:
            signals['semantic'] = float(np.clip(sem_score, 0.0, 1.0))

        # If no signals are available, fall back to existing confidence
        if not signals:
            base_conf = float(det.get('confidence', 0.50))
            return round(float(np.clip(base_conf, 0.01, 0.99)), 4)

        # Normalize weights dynamically for available signals
        active_weights = {k: self.weights.get(k, 0.20) for k in signals}
        w_sum = sum(active_weights.values())
        if w_sum > 0:
            final_conf = sum(active_weights[k] * signals[k] for k in signals) / w_sum
        else:
            final_conf = np.mean(list(signals.values()))

        # Store breakdown in detection for transparency
        det['confidence_signals'] = {k: round(v, 4) for k, v in signals.items()}
        return round(float(np.clip(final_conf, 0.01, 0.99)), 4)

    def _compute_geometry_score(
        self,
        det: Dict[str, Any],
        image: Optional[np.ndarray],
        image_shape: Optional[Tuple[int, int]],
    ) -> Optional[float]:
        """Evaluate spatial density and bounding box shape regularity."""
        bbox = det.get('bbox', {})
        x1, y1 = bbox.get('x_min', 0), bbox.get('y_min', 0)
        x2, y2 = bbox.get('x_max', 0), bbox.get('y_max', 0)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        ar = bh / float(bw)

        cls = det.get('class', 'main_text')

        # Aspect ratio score based on class expectations
        if cls == 'header':
            ar_score = 0.95 if ar < 0.40 else (0.80 if ar < 0.80 else 0.50)
        elif cls == 'footer':
            ar_score = 0.95 if ar < 0.40 else 0.70
        elif cls == 'side_text':
            ar_score = 0.95 if ar > 1.20 else (0.80 if ar > 0.60 else 0.55)
        elif cls == 'main_text':
            ar_score = 0.90 if 0.30 <= ar <= 3.0 else 0.65
        else:  # filler
            ar_score = 0.85

        # Ink density score
        dens = det.get('density')
        if dens is None and image is not None:
            h, w = image.shape[:2]
            cx1, cy1 = max(0, min(w, x1)), max(0, min(h, y1))
            cx2, cy2 = max(0, min(w, x2)), max(0, min(h, y2))
            if cx2 > cx1 and cy2 > cy1:
                roi = image[cy1:cy2, cx1:cx2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
                _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                dens = float(np.count_nonzero(bin_img) / max(1, bin_img.size))
                det['density'] = round(dens, 4)

        if dens is not None:
            # Optimal text density in historical manuscripts is 0.05 to 0.40
            if 0.05 <= dens <= 0.45:
                dens_score = 0.95
            elif dens < 0.02:
                dens_score = 0.25
            elif dens > 0.85:
                dens_score = 0.40
            else:
                dens_score = 0.75
            return 0.50 * ar_score + 0.50 * dens_score

        return ar_score

    def _compute_semantic_score(self, det: Dict[str, Any]) -> Optional[float]:
        """Evaluate script consistency with assignment class definition."""
        cls = det.get('class', 'main_text')
        deva_ratio = det.get('deva_ratio', det.get('devanagari_ratio'))
        latin_ratio = det.get('latin_ratio')

        if deva_ratio is None and latin_ratio is None:
            return None

        deva = float(deva_ratio or 0.0)
        latin = float(latin_ratio or 0.0)

        # Manuscript core classes (main_text, side_text, header, footer) expect Devanagari text
        if cls in {'main_text', 'side_text', 'header', 'footer'}:
            if deva >= 0.75:
                return 0.95
            elif deva >= 0.40:
                return 0.80
            elif latin >= 0.60:
                # Latin text in core manuscript body is suspicious
                return 0.30
            else:
                return 0.65

        # Filler expects English/Latin attribution, pencil notes, or decorative non-text
        if cls == 'filler':
            if latin >= 0.60:
                return 0.98
            elif latin >= 0.30:
                return 0.85
            elif deva >= 0.75:
                # Authentic Devanagari text marked as filler is suspicious
                return 0.25
            else:
                # Decorative elements / seals without text
                return 0.80

        return 0.75
