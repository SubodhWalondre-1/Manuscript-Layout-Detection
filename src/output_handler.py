"""
Output management — annotated images + JSON metadata files.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np

from .utils import ensure_dir

logger = logging.getLogger('manuscript-detector')

MODEL_VERSION = "pp-doclayoutv2-manuscript-hybrid-v1.0"


def save_annotated_image(image: np.ndarray,
                          output_dir: str,
                          filename: str) -> str:
    """Save annotated image to results/annotated/."""
    d = ensure_dir(os.path.join(output_dir, 'annotated'))
    stem = Path(filename).stem
    out = os.path.join(str(d), f"{stem}_annotated.png")
    cv2.imwrite(out, image)
    logger.debug(f"Saved annotated: {out}")
    return out


def save_json_metadata(detections: List[Dict[str, Any]],
                        image_name: str,
                        image_size: Tuple[int, int],
                        output_dir: str,
                        processing_time: float) -> str:
    """Save JSON metadata to results/metadata/."""
    d = ensure_dir(os.path.join(output_dir, 'metadata'))
    stem = Path(image_name).stem
    out = os.path.join(str(d), f"{stem}_metadata.json")

    allowed_classes = {'header', 'footer', 'main_text', 'side_text', 'filler'}

    numbered = []
    for det in detections:
        cls = det.get('class')
        if cls not in allowed_classes:
            logger.warning(f"Detection with invalid class '{cls}' filtered from output JSON")
            continue
        entry = {
            'id': len(numbered) + 1,
            'label': cls,
            'confidence': round(det['confidence'], 4),
            'bbox': {k: int(v) for k, v in det['bbox'].items()},
        }
        if det.get('ocr_text'):
            entry['ocr_text'] = det['ocr_text']
            entry['ocr_confidence'] = round(float(det.get('ocr_conf', 0.0)), 4)
            entry['devanagari_ratio'] = round(float(det.get('deva_ratio', 0.0)), 4)
            entry['latin_ratio'] = round(float(det.get('latin_ratio', 0.0)), 4)
        numbered.append(entry)

    meta = {
        'image_name': image_name,
        'image_size': {'width': image_size[0], 'height': image_size[1]},
        'detection_count': len(numbered),
        'detections': numbered,
        'processing_time_seconds': round(processing_time, 3),
        'model_version': MODEL_VERSION,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }

    with open(out, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.debug(f"Saved JSON: {out}")
    return out


def print_summary(total_images: int, total_dets: int,
                   total_time: float, output_dir: str) -> None:
    avg_time = (total_time / total_images) if total_images else 0.0
    sep = "-" * 60
    logger.info(sep)
    logger.info(
        f"Processed: {total_images} | Detections: {total_dets} | "
        f"Total time: {total_time:.1f}s | Avg/image: {avg_time:.1f}s | Output: {output_dir}"
    )
    logger.info(sep)
