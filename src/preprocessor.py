"""
Image preprocessing pipeline for historical manuscript images.

Handles image loading, EXIF orientation, deskewing, CLAHE enhancement,
optional denoising, and aspect-ratio-preserving resize with full
transformation tracking for coordinate restoration.
"""

import logging
from typing import Tuple, Optional, Dict, List, Any

import cv2
import numpy as np
from PIL import Image, ExifTags

logger = logging.getLogger('manuscript-detector')


def load_image(path: str) -> np.ndarray:
    """Load and validate a BGR image from disk."""
    img = cv2.imread(path)
    if img is None or img.size == 0:
        raise ValueError(f"Failed to load image (corrupt or unsupported): {path}")
    logger.debug(f"Loaded: {path} ({img.shape[1]}x{img.shape[0]})")
    return img


def fix_orientation(image: np.ndarray, path: str) -> np.ndarray:
    """Auto-rotate based on EXIF orientation tag."""
    try:
        pil = Image.open(path)
        exif = pil.getexif()
        if not exif:
            return image
        orient_key = next(
            (k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None
        )
        if orient_key is None or orient_key not in exif:
            return image
        o = exif[orient_key]
        if o == 3:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif o == 6:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif o == 8:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception as e:
        logger.debug(f"EXIF check skipped: {e}")
    return image


def deskew(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Correct page skew via Hough line analysis (threshold > 1°)."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100,
            minLineLength=image.shape[1] // 4, maxLineGap=20,
        )
        if lines is None:
            return image

        angles = []
        for line in lines:
            try:
                pts = line.flatten()
                if len(pts) < 4:
                    continue
                x1, y1, x2, y2 = (
                    int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
                )
                if x2 == x1:
                    continue
                a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if abs(a) < max_angle:
                    angles.append(a)
            except (IndexError, TypeError, ValueError):
                continue

        if not angles:
            return image
        median = float(np.median(angles))
        if abs(median) < 1.0 or abs(median) > max_angle:
            return image

        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), median, 1.0)
        return cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except Exception as e:
        logger.warning(f"Deskew failed: {e}")
        return image


def enhance(image: np.ndarray,
            clip_limit: float = 2.0,
            grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE on the L-channel (LAB) for degraded documents."""
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(grid_size, grid_size)
        )
        lab = cv2.merge([clahe.apply(l), a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        logger.warning(f"CLAHE failed: {e}")
        return image


def denoise(image: np.ndarray) -> np.ndarray:
    """Light non-local means denoising (optional, can be slow)."""
    try:
        return cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
    except Exception as e:
        logger.warning(f"Denoise failed: {e}")
        return image


def letterbox_resize(image: np.ndarray,
                     target_size: int = 1024) -> Tuple[np.ndarray, Dict]:
    """
    Resize with letterboxing to keep aspect ratio.

    Returns (resized_image, transform_info) where transform_info
    contains everything needed to map coordinates back.
    """
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pw, ph = target_size - nw, target_size - nh
    top, left = ph // 2, pw // 2
    padded = cv2.copyMakeBorder(
        resized, top, ph - top, left, pw - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, {
        'scale': scale,
        'pad_left': left,
        'pad_top': top,
        'original_width': w,
        'original_height': h,
    }


def restore_bbox(bbox: Dict[str, Any], transform_info: Dict[str, Any]) -> Dict[str, int]:
    """
    Restore bounding box coordinates from letterboxed/transformed space
    back to the original image coordinate space.

    Args:
        bbox: Dictionary with keys 'x_min', 'y_min', 'x_max', 'y_max'.
        transform_info: Dictionary containing 'scale', 'pad_left', 'pad_top',
                        'original_width', 'original_height'.

    Returns:
        Restored bounding box dictionary with integer coordinates clamped to original image dimensions.
    """
    scale = transform_info.get('scale', 1.0)
    pad_left = transform_info.get('pad_left', 0)
    pad_top = transform_info.get('pad_top', 0)
    orig_w = transform_info.get('original_width')
    orig_h = transform_info.get('original_height')

    if scale <= 0:
        scale = 1.0

    x_min = (bbox['x_min'] - pad_left) / scale
    y_min = (bbox['y_min'] - pad_top) / scale
    x_max = (bbox['x_max'] - pad_left) / scale
    y_max = (bbox['y_max'] - pad_top) / scale

    rx_min = int(round(x_min))
    ry_min = int(round(y_min))
    rx_max = int(round(x_max))
    ry_max = int(round(y_max))

    if orig_w is not None and orig_h is not None:
        rx_min = max(0, min(orig_w - 1, rx_min))
        ry_min = max(0, min(orig_h - 1, ry_min))
        rx_max = max(rx_min + 1, min(orig_w, rx_max))
        ry_max = max(ry_min + 1, min(orig_h, ry_max))

    return {
        'x_min': rx_min,
        'y_min': ry_min,
        'x_max': rx_max,
        'y_max': ry_max,
    }


def restore_detections(detections: List[Dict[str, Any]],
                       transform_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Restore bounding boxes of a list of detections from transformed coordinate space
    to original image coordinates.
    """
    if not transform_info or (
        transform_info.get('scale', 1.0) == 1.0
        and transform_info.get('pad_left', 0) == 0
        and transform_info.get('pad_top', 0) == 0
    ):
        return detections

    restored = []
    for d in detections:
        item = dict(d)
        if 'bbox' in item:
            item['bbox'] = restore_bbox(item['bbox'], transform_info)
        restored.append(item)
    return restored


def preprocess(image_path: str,
               config: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Modular preprocessing pipeline for manuscript images.

    Preserves the original image unchanged and tracks transformation metadata
    so any downstream bounding boxes can be accurately mapped back via restore_detections().

    Returns:
        (processed_image, original_image, transform_info)
    """
    cfg = (config or {}).get('preprocessing', {})

    # Load original image (strictly read-only)
    image = load_image(image_path)
    original = image.copy()
    h, w = original.shape[:2]

    # Check and apply EXIF orientation if metadata indicates rotation
    image = fix_orientation(image, image_path)

    if cfg.get('deskew', False):
        image = deskew(image, cfg.get('deskew_max_angle', 15))
    if cfg.get('enhance', False):
        image = enhance(
            image,
            cfg.get('clahe_clip_limit', 2.0),
            cfg.get('clahe_grid_size', 8),
        )
    if cfg.get('denoise', False):
        image = denoise(image)

    if cfg.get('letterbox', False) or cfg.get('resize', False):
        processed, transform = letterbox_resize(
            image, cfg.get('max_size', 1024)
        )
    else:
        processed = image
        transform = {
            'scale': 1.0,
            'pad_left': 0,
            'pad_top': 0,
            'original_width': w,
            'original_height': h,
        }

    logger.debug(
        f"Preprocessed: {image_path} | "
        f"{original.shape[1]}x{original.shape[0]} -> "
        f"{processed.shape[1]}x{processed.shape[0]}"
    )
    return processed, original, transform
