"""
Post-processing: clipping, confidence filtering, area filtering,
class-aware NMS, overlapping-box merging, and invalid-detection removal.
"""

import logging
from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger('manuscript-detector')


def clip_boxes(dets: List[Dict[str, Any]],
               w: int, h: int) -> List[Dict[str, Any]]:
    """Clamp every box to stay inside image boundaries."""
    out = []
    for d in dets:
        d = deepcopy(d)
        b = d['bbox']
        b['x_min'] = max(0, b['x_min'])
        b['y_min'] = max(0, b['y_min'])
        b['x_max'] = min(w, b['x_max'])
        b['y_max'] = min(h, b['y_max'])
        if b['x_min'] < b['x_max'] and b['y_min'] < b['y_max']:
            out.append(d)
    return out


def filter_confidence(dets: List[Dict[str, Any]],
                       thr: float = 0.10) -> List[Dict[str, Any]]:
    return [d for d in dets if d['confidence'] >= thr]


def filter_area(dets: List[Dict[str, Any]],
                img_area: int,
                min_ratio: float = 0.003,
                min_header_ratio: Optional[float] = None,
                min_footer_ratio: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Remove boxes smaller than min_ratio of image area.
    Header regions (folio numbers, running headers, section titles) use
    min_header_ratio (defaults to min(min_ratio, 0.0006)) to retain valid,
    compact margin text.
    Footer regions (page folios, catchwords, colophons) use
    min_footer_ratio (defaults to min(min_ratio, 0.001)) to retain valid,
    compact footer markers.
    """
    min_a = img_area * min_ratio
    hdr_ratio = min_header_ratio if min_header_ratio is not None else min(min_ratio, 0.0006)
    ftr_ratio = min_footer_ratio if min_footer_ratio is not None else min(min_ratio, 0.001)
    min_a_hdr = img_area * hdr_ratio
    min_a_ftr = img_area * ftr_ratio
    return [
        d for d in dets
        if (d['bbox']['x_max'] - d['bbox']['x_min'])
         * (d['bbox']['y_max'] - d['bbox']['y_min']) >= (
             min_a_hdr if d.get('class') == 'header' else (
                 min_a_ftr if d.get('class') == 'footer' else min_a
             )
         )
    ]


def _iou(a: Dict, b: Dict) -> float:
    x1 = max(a['x_min'], b['x_min'])
    y1 = max(a['y_min'], b['y_min'])
    x2 = min(a['x_max'], b['x_max'])
    y2 = min(a['y_max'], b['y_max'])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a['x_max'] - a['x_min']) * (a['y_max'] - a['y_min'])
    ab = (b['x_max'] - b['x_min']) * (b['y_max'] - b['y_min'])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def apply_nms(dets: List[Dict[str, Any]],
               iou_thr: float = 0.45) -> List[Dict[str, Any]]:
    """Class-aware non-maximum suppression."""
    groups: Dict[str, List] = {}
    for d in dets:
        groups.setdefault(d['class'], []).append(d)

    result = []
    for cls, grp in groups.items():
        grp.sort(key=lambda x: x['confidence'], reverse=True)
        keep = []
        while grp:
            best = grp.pop(0)
            keep.append(best)
            grp = [d for d in grp if _iou(best['bbox'], d['bbox']) < iou_thr]
        result.extend(keep)
    return result


def merge_overlapping(dets: List[Dict[str, Any]],
                       iou_thr: float = 0.65) -> List[Dict[str, Any]]:
    """Merge heavily overlapping same-class boxes."""
    groups: Dict[str, List] = {}
    for d in dets:
        groups.setdefault(d['class'], []).append(d)

    result = []
    for cls, grp in groups.items():
        merged = set()
        for i in range(len(grp)):
            if i in merged:
                continue
            cur = deepcopy(grp[i])
            for j in range(i + 1, len(grp)):
                if j in merged:
                    continue
                should_merge = (_iou(cur['bbox'], grp[j]['bbox']) >= iou_thr)
                # For header lines in the top margin zone, merge if they vertically overlap substantially or are adjacent lines
                if not should_merge and cls == 'header':
                    b1 = cur['bbox']
                    b2 = grp[j]['bbox']
                    y_overlap = min(b1['y_max'], b2['y_max']) - max(b1['y_min'], b2['y_min'])
                    min_h = min(b1['y_max'] - b1['y_min'], b2['y_max'] - b2['y_min'])
                    y_gap = max(0, max(b1['y_min'], b2['y_min']) - min(b1['y_max'], b2['y_max']))
                    x_overlap = min(b1['x_max'], b2['x_max']) - max(b1['x_min'], b2['x_min'])
                    min_w = min(b1['x_max'] - b1['x_min'], b2['x_max'] - b2['x_min'])
                    if (y_overlap / max(1, min_h) > 0.40 or y_gap <= 25) and (x_overlap / max(1, min_w) > 0.50):
                        should_merge = True

                # For filler lines in the margin zones, merge adjacent or overlapping lines belonging to the same attribution block
                if not should_merge and cls == 'filler':
                    b1 = cur['bbox']
                    b2 = grp[j]['bbox']
                    y_overlap = min(b1['y_max'], b2['y_max']) - max(b1['y_min'], b2['y_min'])
                    min_h = min(b1['y_max'] - b1['y_min'], b2['y_max'] - b2['y_min'])
                    y_gap = max(0, max(b1['y_min'], b2['y_min']) - min(b1['y_max'], b2['y_max']))
                    x_overlap = min(b1['x_max'], b2['x_max']) - max(b1['x_min'], b2['x_min'])
                    min_w = min(b1['x_max'] - b1['x_min'], b2['x_max'] - b2['x_min'])
                    if (y_overlap / max(1, min_h) > 0.30 or y_gap <= 30) and (x_overlap / max(1, min_w) > 0.25):
                        should_merge = True

                if should_merge:
                    cur['bbox']['x_min'] = min(cur['bbox']['x_min'],
                                               grp[j]['bbox']['x_min'])
                    cur['bbox']['y_min'] = min(cur['bbox']['y_min'],
                                               grp[j]['bbox']['y_min'])
                    cur['bbox']['x_max'] = max(cur['bbox']['x_max'],
                                               grp[j]['bbox']['x_max'])
                    cur['bbox']['y_max'] = max(cur['bbox']['y_max'],
                                               grp[j]['bbox']['y_max'])
                    cur['confidence'] = max(cur['confidence'],
                                           grp[j]['confidence'])
                    merged.add(j)
            result.append(cur)
    return result


def tighten_bbox_to_content(image: np.ndarray,
                            bbox: Dict[str, int],
                            pad: int = 3,
                            text_boxes: Optional[List[Dict[str, Any]]] = None) -> Tuple[Dict[str, int], float]:
    """
    Refines and tightens a bounding box to enclose only actual foreground ink/content,
    minimizing unnecessary background area.

    1. If fine-grained OCR text boxes are present inside the candidate, bounds them tightly.
    2. Within that content envelope, extracts foreground ink contours (filtering ruling lines,
       page border artifacts, and substrate noise).
    3. Tightens to the exact bounding rectangle enclosing foreground characters.
    4. Adds configurable padding (default 3px) and clips to image boundaries.

    Returns:
        (tightened_bbox, content_density)
    """
    h, w = image.shape[:2]
    x1, y1 = max(0, int(bbox['x_min'])), max(0, int(bbox['y_min']))
    x2, y2 = min(w, int(bbox['x_max'])), min(h, int(bbox['y_max']))
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return {'x_min': x1, 'y_min': y1, 'x_max': x2, 'y_max': y2}, 0.0

    # Match fine-grained text boxes whose center lies inside candidate
    matched_tbs = []
    if text_boxes:
        for tb in text_boxes:
            cx = (tb['x_min'] + tb['x_max']) / 2.0
            cy = (tb['y_min'] + tb['y_max']) / 2.0
            if (x1 - 6 <= cx <= x2 + 6) and (y1 - 6 <= cy <= y2 + 6):
                matched_tbs.append(tb)

    if matched_tbs:
        tb_x1 = max(0, min(tb['x_min'] for tb in matched_tbs))
        tb_y1 = max(0, min(tb['y_min'] for tb in matched_tbs))
        tb_x2 = min(w, max(tb['x_max'] for tb in matched_tbs))
        tb_y2 = min(h, max(tb['y_max'] for tb in matched_tbs))
        x1, y1, x2, y2 = tb_x1, tb_y1, tb_x2, tb_y2

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return {'x_min': x1, 'y_min': y1, 'x_max': x2, 'y_max': y2}, 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    rh, rw = gray.shape

    mean_val = np.mean(gray)
    if mean_val < 100:
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, -4
        )
    else:
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 4
        )

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_boxes = []
    for c in contours:
        bx, by, bw_c, bh_c = cv2.boundingRect(c)
        # Exclude ruling lines spanning almost the entire crop
        if (bw_c > 0.80 * rw and bh_c <= 4) or (bh_c > 0.80 * rh and bw_c <= 4):
            continue
        # Exclude crop boundary touch lines
        if (bx == 0 or bx + bw_c == rw) and bw_c <= 2:
            continue
        if (by == 0 or by + bh_c == rh) and bh_c <= 2:
            continue
        # Exclude tiny noise specks
        if cv2.contourArea(c) < 3 and bw_c <= 2 and bh_c <= 2:
            continue
        valid_boxes.append((bx, by, bw_c, bh_c))

    if not valid_boxes:
        return {
            'x_min': max(0, x1 - pad),
            'y_min': max(0, y1 - pad),
            'x_max': min(w, x2 + pad),
            'y_max': min(h, y2 + pad),
        }, 0.0

    # In compact / single-line crops, isolate the dominant vertical ink band
    # to avoid stretching across whitespace gaps into adjacent line descenders/ascenders
    if rh <= 55 and valid_boxes:
        row_counts = np.count_nonzero(bin_img, axis=1)
        peak = int(np.max(row_counts))
        if peak >= 10:
            peak_idx = int(np.argmax(row_counts))
            top_idx = peak_idx
            while top_idx > 0 and row_counts[top_idx - 1] > 0.05 * peak:
                top_idx -= 1
            bot_idx = peak_idx
            while bot_idx < len(row_counts) - 1 and row_counts[bot_idx + 1] > 0.05 * peak:
                bot_idx += 1
            band_boxes = [b for b in valid_boxes if max(b[1], top_idx) <= min(b[1] + b[3], bot_idx + 1)]
            if band_boxes:
                valid_boxes = band_boxes
                pad = min(pad, 2)

    rx1 = min(b[0] for b in valid_boxes)
    ry1 = min(b[1] for b in valid_boxes)
    rx2 = max(b[0] + b[2] for b in valid_boxes)
    ry2 = max(b[1] + b[3] for b in valid_boxes)

    content_pixels = np.count_nonzero(bin_img[ry1:ry2, rx1:rx2])
    area = max(1, (rx2 - rx1) * (ry2 - ry1))
    density = content_pixels / area

    return {
        'x_min': max(0, x1 + rx1 - pad),
        'y_min': max(0, y1 + ry1 - pad),
        'x_max': min(w, x1 + rx2 + pad),
        'y_max': min(h, y1 + ry2 + pad),
    }, float(density)


def split_and_localize_filler(image: np.ndarray,
                              candidate: Dict[str, Any],
                              text_boxes: Optional[List[Dict[str, Any]]] = None,
                              pad: int = 3) -> List[Dict[str, Any]]:
    """
    Tightly encloses actual filler content (English text, pencil text, decorative elements).
    If multiple separated filler items exist (gap > 45px), creates separate tight boxes
    instead of one giant box.
    """
    h, w = image.shape[:2]
    b = candidate['bbox']
    x1, y1 = max(0, int(b['x_min'])), max(0, int(b['y_min']))
    x2, y2 = min(w, int(b['x_max'])), min(h, int(b['y_max']))

    # Check if fine-grained text boxes belong to this filler region
    matched_tbs = candidate.get('text_boxes')
    if not matched_tbs and text_boxes:
        matched_tbs = [
            tb for tb in text_boxes
            if (x1 - 6 <= (tb['x_min'] + tb['x_max']) / 2.0 <= x2 + 6)
            and (y1 - 6 <= (tb['y_min'] + tb['y_max']) / 2.0 <= y2 + 6)
        ]

    if matched_tbs and len(matched_tbs) > 1:
        matched_sorted = sorted(matched_tbs, key=lambda tb: tb['y_min'])
        clusters = []
        cur_cl = [matched_sorted[0]]
        for tb in matched_sorted[1:]:
            if tb['y_min'] - cur_cl[-1]['y_max'] <= 45:
                cur_cl.append(tb)
            else:
                clusters.append(cur_cl)
                cur_cl = [tb]
        clusters.append(cur_cl)

        if len(clusters) > 1:
            split_items = []
            for cl in clusters:
                cl_bbox = {
                    'x_min': min(tb['x_min'] for tb in cl),
                    'y_min': min(tb['y_min'] for tb in cl),
                    'x_max': max(tb['x_max'] for tb in cl),
                    'y_max': max(tb['y_max'] for tb in cl),
                }
                tight_b, dens = tighten_bbox_to_content(image, cl_bbox, pad=pad, text_boxes=cl)
                item = deepcopy(candidate)
                item['bbox'] = tight_b
                item['density'] = round(dens, 4)
                item['text_boxes'] = cl
                split_items.append(item)
            return split_items

    # Single item or pixel-based tightening
    tight_b, dens = tighten_bbox_to_content(image, b, pad=pad, text_boxes=matched_tbs)
    item = deepcopy(candidate)
    item['bbox'] = tight_b
    item['density'] = round(dens, 4)
    return [item]


TARGET_CLASSES = {'header', 'main_text', 'footer', 'side_text', 'filler'}


def suppress_container_boxes(dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    If a large filler box contains a much smaller filler text box,
    suppress the large parent box when most of its area is background.
    """
    to_remove = set()
    n = len(dets)
    for i in range(n):
        for j in range(n):
            if i == j or i in to_remove or j in to_remove:
                continue
            a, b = dets[i], dets[j]
            ba, bb = a['bbox'], b['bbox']
            area_a = (ba['x_max'] - ba['x_min']) * (ba['y_max'] - ba['y_min'])
            area_b = (bb['x_max'] - bb['x_min']) * (bb['y_max'] - bb['y_min'])

            ix1 = max(ba['x_min'], bb['x_min'])
            iy1 = max(ba['y_min'], bb['y_min'])
            ix2 = min(ba['x_max'], bb['x_max'])
            iy2 = min(ba['y_max'], bb['y_max'])

            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                # If b is substantially contained in a, and a is much larger container of the same class
                if a['class'] == b['class'] and a['class'] in ['filler', 'footer', 'header'] and inter / max(1, area_b) > 0.70 and area_b < 0.65 * area_a:
                    logger.debug(
                        f"Suppressing large container box {ba} in favor of tight box {bb}"
                    )
                    to_remove.add(i)
                # If small box b is contained inside larger main_text box a (e.g. palm-leaf binding hole or sub-box artifact)
                elif a['class'] == 'main_text' and b['class'] == 'main_text' and inter / max(1, area_b) > 0.75 and area_b < 0.20 * area_a:
                    logger.debug(
                        f"Suppressing interior substrate artifact {bb} inside main_text parent {ba}"
                    )
                    to_remove.add(j)

    # Suppress giant parent main_text containers that engulf multiple child regions of distinct classes
    # (e.g. macro container engulfing side_text + header/footer or another body block)
    main_count = sum(1 for d in dets if d['class'] == 'main_text')
    for i in range(n):
        if i in to_remove or dets[i]['class'] != 'main_text':
            continue
        ba = dets[i]['bbox']
        area_a = (ba['x_max'] - ba['x_min']) * (ba['y_max'] - ba['y_min'])
        child_classes = set()
        for j in range(n):
            if i == j or j in to_remove:
                continue
            b = dets[j]
            if b['class'] in ['header', 'side_text', 'footer', 'main_text']:
                bb = b['bbox']
                area_b = (bb['x_max'] - bb['x_min']) * (bb['y_max'] - bb['y_min'])
                ix1 = max(ba['x_min'], bb['x_min'])
                iy1 = max(ba['y_min'], bb['y_min'])
                ix2 = min(ba['x_max'], bb['x_max'])
                iy2 = min(ba['y_max'], bb['y_max'])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    if inter / max(1, area_b) > 0.65 and area_b < 0.60 * area_a:
                        child_classes.add(b['class'])
        if len(child_classes) >= 2 and ('side_text' in child_classes or len(child_classes) >= 3 or main_count > 1):
            logger.debug(f"Suppressing giant parent main_text container {ba} containing {child_classes}")
            to_remove.add(i)

    return [d for idx, d in enumerate(dets) if idx not in to_remove]


def correct_or_split_main_text_with_filler(main_dets: List[Dict[str, Any]],
                                            filler_dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensures that NO filler region remains inside or overlaps with main_text.
    If a filler region overlaps or is inside main_text:
    - Clamps main_text boundary if filler is near the edge (top/bottom)
    - Splits main_text into two independent main_text blocks if filler is in the middle
    - Rejects main_text sub-blocks if they have negligible height (< 15px)
    """
    if not filler_dets or not main_dets:
        return main_dets

    current_mains = main_dets
    for f in filler_dets:
        fb = f['bbox']
        fx1, fy1, fx2, fy2 = fb['x_min'], fb['y_min'], fb['x_max'], fb['y_max']
        next_mains = []
        for m in current_mains:
            # A line-owned body has already excluded filler one line at a time.
            # Clamp edge overlap without erasing retained interior lines.
            if m.get('geometry_source') == 'line_owned':
                mb = m['bbox']
                ix1 = max(mb['x_min'], fx1)
                iy1 = max(mb['y_min'], fy1)
                ix2 = min(mb['x_max'], fx2)
                iy2 = min(mb['y_max'], fy2)
                if ix2 > ix1 and iy2 > iy1:
                    m_cy = (mb['y_min'] + mb['y_max']) / 2.0
                    f_cy = (fy1 + fy2) / 2.0
                    m_copy = deepcopy(m)
                    if f_cy > m_cy and mb['y_max'] >= fy1:
                        m_copy['bbox']['y_max'] = max(mb['y_min'] + 10, fy1 - 1)
                    elif f_cy < m_cy and mb['y_min'] <= fy2:
                        m_copy['bbox']['y_min'] = min(mb['y_max'] - 10, fy2 + 1)
                    next_mains.append(m_copy)
                else:
                    next_mains.append(m)
                continue
            mb = m['bbox']
            mx1, my1, mx2, my2 = mb['x_min'], mb['y_min'], mb['x_max'], mb['y_max']

            ix1 = max(mx1, fx1)
            iy1 = max(my1, fy1)
            ix2 = min(mx2, fx2)
            iy2 = min(my2, fy2)

            # If no intersection, retain m as-is
            if ix2 <= ix1 or iy2 <= iy1:
                next_mains.append(m)
                continue

            mh = my2 - my1
            # Case 1: Filler completely covers main_text vertically
            if fy1 <= my1 and fy2 >= my2:
                # Main text is entirely covered by filler; drop it
                continue

            # Case 2: Filler at or near the top of main_text
            if fy1 <= my1 + 0.25 * mh and fy2 < my2:
                new_top = fy2 + 1
                if my2 - new_top >= 15:
                    m_copy = deepcopy(m)
                    m_copy['bbox']['y_min'] = new_top
                    next_mains.append(m_copy)

            # Case 3: Filler at or near the bottom of main_text
            elif fy2 >= my2 - 0.25 * mh and fy1 > my1:
                new_bot = fy1 - 1
                if new_bot - my1 >= 15:
                    m_copy = deepcopy(m)
                    m_copy['bbox']['y_max'] = new_bot
                    next_mains.append(m_copy)

            # Case 4: Filler in the interior / middle of main_text -> SPLIT
            else:
                top_h = fy1 - 1 - my1
                bot_h = my2 - (fy2 + 1)
                if top_h >= 15:
                    m_top = deepcopy(m)
                    m_top['bbox']['y_max'] = fy1 - 1
                    next_mains.append(m_top)
                if bot_h >= 15:
                    m_bot = deepcopy(m)
                    m_bot['bbox']['y_min'] = fy2 + 1
                    next_mains.append(m_bot)

        current_mains = next_mains

    return current_mains


def resolve_conflicts(dets: List[Dict[str, Any]], w: int, h: int) -> List[Dict[str, Any]]:
    """
    Final conflict resolution executed BEFORE bounding box tightening:
    1. Container suppression: removes parent containers that engulf fine-grained regions.
    2. Spurious cross-class duplicate resolution: when detector and OCR fusion produce
       conflicting duplicate predictions for the same content area.
    3. Boundary clamping: ensures disjoint vertical bands do not overlap.
    4. Filler vs main_text splitting/correction: ensures filler never remains inside main_text.
    """
    dets = suppress_container_boxes(dets)

    # Cross-class duplicate suppression:
    # If two boxes of different classes substantially overlap (IoU > 0.45),
    # they represent conflicting duplicate labels for the same physical region.
    # Keep the more specific semantic label or the one with higher confidence.
    n = len(dets)
    to_remove = set()
    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            a, b = dets[i], dets[j]
            if a['class'] == b['class']:
                continue

            iou = _iou(a['bbox'], b['bbox'])
            if iou > 0.45:
                # Specific semantic classes (filler, header, footer, side_text) beat generic main_text
                if a['class'] == 'main_text' and b['class'] in {'filler', 'header', 'footer', 'side_text'}:
                    to_remove.add(i)
                    break
                elif b['class'] == 'main_text' and a['class'] in {'filler', 'header', 'footer', 'side_text'}:
                    to_remove.add(j)
                else:
                    if a.get('confidence', 0) >= b.get('confidence', 0):
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break

    cleaned = [d for idx, d in enumerate(dets) if idx not in to_remove]

    # Vertical band non-overlap clamping:
    headers = [d for d in cleaned if d['class'] == 'header']
    footers = [d for d in cleaned if d['class'] == 'footer']
    bodies = [d for d in cleaned if d['class'] in {'main_text', 'side_text'}]

    if headers and bodies:
        max_hdr_bottom = max(h_det['bbox']['y_max'] for h_det in headers)
        for b in bodies:
            if b.get('geometry_source') != 'line_owned' and b['bbox']['y_min'] < max_hdr_bottom:
                b['bbox']['y_min'] = max_hdr_bottom

    if footers and bodies:
        min_ftr_top = min(f_det['bbox']['y_min'] for f_det in footers)
        for b in bodies:
            if b.get('geometry_source') != 'line_owned' and b['bbox']['y_max'] > min_ftr_top:
                b['bbox']['y_max'] = min_ftr_top

    # Filler vs main_text: ensure no filler remains inside or overlaps main_text
    fillers = [d for d in cleaned if d['class'] == 'filler']
    mains = [d for d in cleaned if d['class'] == 'main_text']
    others = [d for d in cleaned if d['class'] not in {'filler', 'main_text'}]
    if fillers and mains:
        mains = correct_or_split_main_text_with_filler(mains, fillers)
    cleaned = others + fillers + mains

    return cleaned



def _semantic_hallucination_guard(dets: List[Dict[str, Any]], w: int, h: int) -> List[Dict[str, Any]]:
    """Drop oversized macro text containers and impossible footer claims.

    Macro priors are useful as evidence, but a filler/footer proposal that
    overlaps the manuscript body is not a valid final semantic region.
    Line-owned regions remain authoritative because they are backed by the
    fine-grained text inventory.
    """
    mains = [d for d in dets if d.get('class') == 'main_text']
    page_area = max(1, w * h)
    kept = []

    def iou_like_overlap(a, b):
        ix1 = max(a['x_min'], b['x_min'])
        iy1 = max(a['y_min'], b['y_min'])
        ix2 = min(a['x_max'], b['x_max'])
        iy2 = min(a['y_max'], b['y_max'])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1, (a['x_max'] - a['x_min']) * (a['y_max'] - a['y_min']))
        return inter / area_a

    for d in dets:
        cls = d.get('class')
        b = d.get('bbox', {})
        bw = max(0, b.get('x_max', 0) - b.get('x_min', 0))
        bh = max(0, b.get('y_max', 0) - b.get('y_min', 0))
        area = bw * bh
        source = d.get('geometry_source', '')
        overlap_main = max((iou_like_overlap(b, m['bbox']) for m in mains), default=0.0)

        if cls == 'filler':
            # A large macro-prior filler is almost always a detector container
            # when it overlaps the OCR-owned manuscript body. Small English/
            # attribution boxes are still allowed. True visual fillers use the
            # explicit macro_visual source and are handled separately.
            if source not in {'line_owned', 'macro_visual'}:
                if area > 0.20 * page_area:
                    logger.debug(f"Dropping oversized macro filler: {b}")
                    continue
                if overlap_main >= 0.20:
                    logger.debug(f"Dropping filler overlapping main_text: {b}")
                    continue

        if cls == 'footer':
            # Footer must be a compact region genuinely below the body. A macro
            # footer overlapping a body container is a hallucination, not a footer.
            if mains:
                body_y2 = max(m['bbox']['y_max'] for m in mains)
                vertical_overlap = max(0, min(b['y_max'], body_y2) - max(b['y_min'], min(m['bbox']['y_min'] for m in mains)))
                overlap_ratio = vertical_overlap / max(1, bh)
                if b['y_min'] <= body_y2 and overlap_ratio > 0.12:
                    logger.debug(f"Dropping footer intersecting main_text: {b}")
                    continue
                if b['y_min'] < body_y2 and source != 'line_owned':
                    logger.debug(f"Dropping non-line-owned footer inside body band: {b}")
                    continue
            # Reject very tall footer containers even when no main box survived.
            if source != 'line_owned' and bh > 0.08 * h:
                logger.debug(f"Dropping oversized macro footer: {b}")
                continue

        kept.append(d)
    return kept

def postprocess(dets: List[Dict[str, Any]],
                w: int, h: int,
                confidence: float = 0.10,
                config: Optional[dict] = None,
                image: Optional[np.ndarray] = None,
                text_boxes: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Executes the required post-processing pipeline order:
    1. Candidate discovery (completed upstream)
    2. Semantic classification (completed upstream)
    3. Region grouping (completed upstream)
    4. Conflict resolution (BEFORE bbox tightening)
    5. Class-aware NMS & Merge (candidate consolidation)
    6. FINAL BBOX LOCALIZATION (Separate from Region Detection):
       - Derives tight bboxes strictly from actual foreground ink pixels / text polygons
       - Splits multi-item filler into separate tight boxes if separated by gaps
       - Adds small configurable padding (default 3px)
       - Never uses layout candidate boxes as final boxes
    7. Clipping (strictly to [0, w] x [0, h] with invalid box pruning)
    8. Output validation (strictly the 5 target classes, no zero/negative area, inside boundaries)
    """
    pp = (config or {}).get('postprocessing', {})
    nms_iou = pp.get('nms_iou_threshold', 0.45)
    merge_iou = pp.get('merge_iou_threshold', 0.65)
    min_area = pp.get('min_area_ratio', 0.003)
    min_header_area = pp.get('min_header_area_ratio', 0.0006)
    min_footer_area = pp.get('min_footer_area_ratio', 0.001)
    min_conf = max(confidence, pp.get('min_confidence', 0.10))

    # STAGE 4: Conflict Resolution (BEFORE bbox tightening)
    dets = resolve_conflicts(dets, w, h)

    # STAGE 5: Class-aware NMS & Merge (candidate consolidation)
    # Eliminates duplicate overlapping boxes from detector/OCR fusion
    # Legitimate multiple boxes of the same class (with low IoU) are preserved
    # Different semantic classes are NEVER merged
    dets = apply_nms(dets, nms_iou)
    dets = merge_overlapping(dets, merge_iou)

    # STAGE 6: FINAL BBOX LOCALIZATION (Strictly separated from Region Detection)
    # Never use the candidate region's full bounding box as the final output box.
    # Final bbox must be derived from the actual pixels/OCR polygons belonging to that semantic region.
    from src.confidence_scorer import ConfidenceScorer
    scorer = ConfidenceScorer(config)

    localized = []
    for d in dets:
        d = deepcopy(d)
        if image is not None:
            # Handle multi-item splitting for filler if separated items exist
            if d['class'] == 'filler':
                split_items = split_and_localize_filler(image, d, text_boxes=text_boxes, pad=3)
                for item in split_items:
                    dens = item.get('density', 0.1)
                    if dens < 0.03:
                        logger.debug(f"Rejected low-density filler item: {item['bbox']} (dens={dens:.3f})")
                        continue
                    item['confidence'] = scorer.compute_confidence(item, image, (h, w))
                    localized.append(item)
            else:
                tight_b, dens = tighten_bbox_to_content(image, d['bbox'], pad=3, text_boxes=text_boxes)
                if tight_b['x_max'] > tight_b['x_min'] and tight_b['y_max'] > tight_b['y_min']:
                    d['bbox'] = tight_b
                d['density'] = round(dens, 4)
                d['confidence'] = scorer.compute_confidence(d, image, (h, w))
                localized.append(d)
        else:
            d['confidence'] = scorer.compute_confidence(d, image, (h, w))
            localized.append(d)
    # Remove semantically impossible macro containers before overlap splitting.
    localized = _semantic_hallucination_guard(localized, w, h)

    # Final check: ensure no filler overlaps with main_text after tightening
    fillers_loc = [d for d in localized if d['class'] == 'filler']
    mains_loc = [d for d in localized if d['class'] == 'main_text']
    others_loc = [d for d in localized if d['class'] not in {'filler', 'main_text'}]
    if fillers_loc and mains_loc:
        mains_loc = correct_or_split_main_text_with_filler(mains_loc, fillers_loc)
    dets = others_loc + fillers_loc + mains_loc

    # Ensure mutually exclusive semantic column/band boundaries do not overlap
    headers = [d for d in dets if d['class'] == 'header']
    footers = [d for d in dets if d['class'] == 'footer']
    mains = [d for d in dets if d['class'] == 'main_text']
    sides = [d for d in dets if d['class'] == 'side_text']
    fillers = [d for d in dets if d['class'] == 'filler']

    if mains:
        min_main_x = min(m['bbox']['x_min'] for m in mains)
        max_main_x = max(m['bbox']['x_max'] for m in mains)
        min_main_y = min(m['bbox']['y_min'] for m in mains)
        max_main_y = max(m['bbox']['y_max'] for m in mains)

        # Left side_text should be to the left of main_text, right side_text to the right
        for s in sides:
            cx = (s['bbox']['x_min'] + s['bbox']['x_max']) / 2.0
            if cx < min_main_x:
                if s['bbox']['x_max'] > min_main_x:
                    s['bbox']['x_max'] = min_main_x
            elif cx > max_main_x:
                if s['bbox']['x_min'] < max_main_x:
                    s['bbox']['x_min'] = max_main_x

        # Headers should be above main_text
        for hdr in headers:
            if hdr['bbox']['y_max'] > min_main_y:
                hdr['bbox']['y_max'] = min_main_y

        # Footers should be below main_text
        for ftr in footers:
            if ftr['bbox']['y_min'] < max_main_y:
                ftr['bbox']['y_min'] = max_main_y

    if footers and fillers:
        max_ftr_y = max(f['bbox']['y_max'] for f in footers)
        for fil in fillers:
            if (fil['bbox']['y_min'] + fil['bbox']['y_max']) / 2.0 > max_ftr_y:
                if fil['bbox']['y_min'] < max_ftr_y:
                    fil['bbox']['y_min'] = max_ftr_y

    if mains and fillers:
        for m in mains:
            for fil in fillers:
                h_overlap = max(0, min(m['bbox']['x_max'], fil['bbox']['x_max']) -
                                   max(m['bbox']['x_min'], fil['bbox']['x_min']))
                if h_overlap <= 0:
                    continue
                m_cy = (m['bbox']['y_min'] + m['bbox']['y_max']) / 2.0
                f_cy = (fil['bbox']['y_min'] + fil['bbox']['y_max']) / 2.0
                if f_cy > m_cy and m['bbox']['y_max'] >= fil['bbox']['y_min']:
                    m['bbox']['y_max'] = max(m['bbox']['y_min'] + 10, fil['bbox']['y_min'] - 1)
                elif f_cy < m_cy and m['bbox']['y_min'] <= fil['bbox']['y_max']:
                    m['bbox']['y_min'] = min(m['bbox']['y_max'] - 10, fil['bbox']['y_max'] + 1)

    # STAGE 7: Clipping (strictly to [0, w] x [0, h] & impossible coordinate removal)
    dets = clip_boxes(dets, w, h)

    # STAGE 8: Output Validation
    # Confidence and area filtering
    dets = filter_confidence(dets, min_conf)
    dets = filter_area(dets, w * h, min_area, min_header_area, min_footer_area)

    # Strictly validate against assignment target classes and impossible coordinates
    final_output = []
    for d in dets:
        cls = d.get('class')
        if cls not in TARGET_CLASSES:
            continue
        b = d['bbox']
        # Assert valid inside boundary geometry
        if b['x_min'] < 0 or b['y_min'] < 0 or b['x_max'] > w or b['y_max'] > h:
            continue
        if b['x_max'] <= b['x_min'] or b['y_max'] <= b['y_min']:
            continue
        # False main_text elimination: reject main_text boxes with no meaningful ink/text evidence
        if cls == 'main_text' and image is not None:
            dens = d.get('density')
            if dens is None:
                _, dens = tighten_bbox_to_content(image, b)
                d['density'] = round(dens, 4)
            if dens < 0.015 and not d.get('text_boxes') and not d.get('ocr_text', '').strip():
                logger.debug(f"Rejected false main_text in blank/background area: {b} (dens={dens:.4f})")
                continue
        final_output.append(d)

    logger.debug(f"Post-processing complete: {len(final_output)} detections")
    return final_output
