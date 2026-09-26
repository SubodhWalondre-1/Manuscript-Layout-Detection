"""
Annotated image generation with colour-coded bounding boxes.

Always works on a COPY of the original image — never mutates the input.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger('manuscript-detector')

DEFAULT_COLORS = {
    'header': (50, 100, 255),
    'footer': (255, 200, 50),
    'main_text': (100, 255, 100),
    'side_text': (50, 200, 255),
    'filler': (255, 100, 200),
}


def draw_detections(image: np.ndarray,
                     detections: List[Dict[str, Any]],
                     config: Optional[dict] = None) -> np.ndarray:
    """
    Draw bounding boxes + compact labels on a COPY of the image.
    Uses prominent 3–4 px border lines, readable font, and compact label badges
    positioned just above or inside the bounding box without overlapping manuscript content.
    """
    out = image.copy()
    h, w = out.shape[:2]

    vis = (config or {}).get('visualization', {})
    colors_cfg = vis.get('colors', {})
    f_ratio = vis.get('font_scale_ratio', 0.00045)

    color_map = {}
    for cls, default in DEFAULT_COLORS.items():
        c = colors_cfg.get(cls, list(default))
        color_map[cls] = tuple(c) if isinstance(c, list) else default

    base = max(w, h)
    # Bounding-box line width: clean and thin (2 px default, 1 px for small crops)
    default_thick = 2 if base >= 600 else 1
    thick = max(1, int(vis.get('line_thickness', default_thick)))

    # Keep label text clean and clearly readable
    fscale = max(0.36, min(0.48, base * f_ratio))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thick = 1

    for det in detections:
        b = det['bbox']
        cls = det['class']
        conf = det['confidence']
        color = color_map.get(cls, (200, 200, 200))

        x1, y1 = max(0, int(b['x_min'])), max(0, int(b['y_min']))
        x2, y2 = min(w, int(b['x_max'])), min(h, int(b['y_max']))

        # Draw prominent bounding box (3–4 px)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

        # Compact label: class: confidence
        label = f"{cls}: {conf:.2f}"
        (lw, lh), baseline = cv2.getTextSize(
            label, font, fscale, font_thick
        )

        badge_w = lw + 8
        badge_h = lh + baseline + 6

        # Position label: prefer just above bounding box unless another detection is above,
        # otherwise place below (if room in image) or fallback to inside the box
        has_box_above = any(
            other != det and
            other.get('bbox', {}).get('y_max', 0) <= y1 and
            y1 - other.get('bbox', {}).get('y_max', 0) < badge_h + 6 and
            max(0, min(x2, other.get('bbox', {}).get('x_max', 0)) -
                   max(x1, other.get('bbox', {}).get('x_min', 0))) > 0
            for other in detections if isinstance(other, dict) and 'bbox' in other
        )
        if (not has_box_above) and y1 >= badge_h + thick:
            ly1 = y1 - badge_h
            ly2 = y1
        elif y2 + badge_h <= h:
            ly1 = y2
            ly2 = y2 + badge_h
        else:
            ly1 = y1
            ly2 = min(h, y1 + badge_h)

        lx1 = max(0, min(x1, w - badge_w))
        lx2 = min(w, lx1 + badge_w)

        # Background badge with solid contrast against textured parchment
        overlay = out.copy()
        cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.addWeighted(overlay, 0.90, out, 0.10, 0, out)

        brightness = color[2] * 0.299 + color[1] * 0.587 + color[0] * 0.114
        txt_color = (15, 15, 15) if brightness > 128 else (255, 255, 255)
        cv2.putText(
            out, label, (lx1 + 4, ly2 - baseline - 2),
            font, fscale, txt_color, font_thick, cv2.LINE_AA,
        )

    return out
