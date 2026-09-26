"""
Region Reconstruction Engine for Two-Source Region Proposal Architecture.

Implements the assignment-specific layout reconstruction stage:
1. OCR Line / Block Grouping (overlap, gap, height similarity, alignment)
2. Main Text Identification (dominant body columns, coverage, centrality, multiple columns allowed)
3. Symmetric Side Text Discovery (left and right margins, independent of layout model)
4. Header Detection (adaptive top band, separation gap, title-like structure)
5. Independent Footer Detection (derived from lower page geometry, not scanning below oversized box)
6. Multi-Signal Filler Detection (script, visual elements, decorative motifs, faint pencil text)
7. Conflict Resolution & Container Suppression (no swallowing of semantic regions)
8. Multi-Region Support (multiple blocks per class)
9. Final Geometry (tighten, clip, validate)

All operations use normalized geometry and adaptive document-intrinsic statistics.
No fixed pixel coordinates, no hard-coded image dimensions.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger('manuscript-detector')


class RegionReconstructor:
    """
    Reconstructs manuscript layout regions from fine-grained text detection geometry
    and resolves conflicts with PP-DocLayoutV2 macro proposals.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}

    def compute_adaptive_metrics(
        self, text_boxes: List[Dict[str, Any]], image_shape: Tuple[int, int]
    ) -> Dict[str, float]:
        """
        Compute document-intrinsic spatial metrics:
        - Median line height (H_line)
        - Median inter-line vertical gap (G_vert)
        - Median line width (W_line)
        - Aspect ratio (W / H)
        """
        h, w = image_shape
        if not text_boxes:
            return {
                'h_line': max(8.0, h * 0.015),
                'g_vert': max(4.0, h * 0.008),
                'w_line': max(30.0, w * 0.30),
                'aspect_ratio': float(w / max(1, h)),
            }

        heights = [float(b['y_max'] - b['y_min']) for b in text_boxes]
        widths = [float(b['x_max'] - b['x_min']) for b in text_boxes]
        # OCR inputs are normally line boxes, but compatibility callers may
        # supply a few paragraph/column blocks.  Estimate line cadence from the
        # lower-height portion so a single tall block cannot inflate role gaps.
        if heights:
            sorted_heights = sorted(heights)
            pool_n = max(3, int(np.ceil(0.70 * len(sorted_heights))))
            h_line = float(np.median(sorted_heights[:pool_n]))
        else:
            h_line = max(8.0, h * 0.015)
        w_line = float(np.percentile(widths, 75)) if widths else max(30.0, w * 0.30)

        # Estimate vertical cadence per horizontally compatible neighbour.
        # A global y-sort is wrong for marginalia/multi-column pages because a
        # line in another column may sit between two lines of the same flow.
        gaps = []
        ordered = sorted(text_boxes, key=lambda b: (b['y_min'], b['x_min']))
        for box in ordered:
            bw = max(1.0, float(box['x_max'] - box['x_min']))
            cx = 0.5 * (box['x_min'] + box['x_max'])
            best_gap = None
            best_distance = float('inf')
            for other in ordered:
                if other is box:
                    continue
                if other['y_max'] > box['y_min']:
                    continue
                obw = max(1.0, float(other['x_max'] - other['x_min']))
                ocx = 0.5 * (other['x_min'] + other['x_max'])
                union = max(1.0, max(box['x_max'], other['x_max']) - min(box['x_min'], other['x_min']))
                overlap = max(0.0, min(box['x_max'], other['x_max']) - max(box['x_min'], other['x_min'])) / union
                centre_close = abs(cx - ocx) <= 0.30 * max(bw, obw)
                if overlap < 0.18 and not centre_close:
                    continue
                gap = float(box['y_min'] - other['y_max'])
                if 0 < gap < 3.0 * h_line and gap < best_distance:
                    best_gap = gap
                    best_distance = gap
            if best_gap is not None:
                gaps.append(best_gap)

        g_vert = float(np.median(gaps)) if gaps else max(4.0, h_line * 0.5)

        return {
            'h_line': max(6.0, h_line),
            'g_vert': max(3.0, g_vert),
            'w_line': max(20.0, w_line),
            'aspect_ratio': float(w / max(1, h)),
        }

    def analyze_structure(
        self,
        image: np.ndarray,
        text_boxes: List[Dict[str, Any]],
        macro_proposals: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return page-relative structure as evidence, never final class labels.

        Baselines are segmented *before* flow linking.  This is important for
        a framed commentary page: components on the same y coordinate can be
        three different reading flows, not one very wide text line.
        """
        h, w = image.shape[:2]
        valid = [
            b for b in text_boxes
            if b.get('x_max', 0) > b.get('x_min', 0)
            and b.get('y_max', 0) > b.get('y_min', 0)
        ]
        metrics = self.compute_adaptive_metrics(valid, (h, w))
        frame = self._estimate_content_frame(valid, (h, w), metrics)
        separators = self._discover_structure_separators(image, valid, frame, metrics)
        gutters = self._find_column_gutters(valid, w) if len(valid) >= 4 else []
        baseline_segments = self._segment_baselines(valid, frame, separators, metrics)
        flows = self._link_reading_flows(baseline_segments, frame, separators, metrics)
        components = []
        for b in valid:
            components.append({
                'bbox': b,
                'x_center_norm': ((b['x_min'] + b['x_max']) / 2.0) / max(1, w),
                'y_center_norm': ((b['y_min'] + b['y_max']) / 2.0) / max(1, h),
                'width_norm': (b['x_max'] - b['x_min']) / max(1, w),
                'height_norm': (b['y_max'] - b['y_min']) / max(1, h),
            })
        return {
            'page_shape': (h, w),
            'content_frame': frame,
            'separators': separators,
            'baseline_segments': baseline_segments,
            'flows': flows,
            'metrics': {
                **metrics,
                'line_height_norm': metrics['h_line'] / max(1, h),
                'line_gap_norm': metrics['g_vert'] / max(1, h),
            },
            'gutters_norm': [g / max(1, w) for g in gutters],
            'components': components,
            'macro_proposals': list(macro_proposals),
        }

    @staticmethod
    def _bbox_from_boxes(boxes: List[Dict[str, Any]]) -> Dict[str, int]:
        return {
            'x_min': int(min(box['x_min'] for box in boxes)),
            'y_min': int(min(box['y_min'] for box in boxes)),
            'x_max': int(max(box['x_max'] for box in boxes)),
            'y_max': int(max(box['y_max'] for box in boxes)),
        }

    def _estimate_content_frame(self, boxes: List[Dict[str, Any]], image_shape: Tuple[int, int],
                                metrics: Dict[str, float]) -> Dict[str, int]:
        """Estimate a reusable text-bearing frame from robust component bounds.

        Percentiles avoid letting a small external attribution expand the
        manuscript frame to the entire scan.  The frame is evidence only; no
        final detection is ever emitted from it.
        """
        h, w = image_shape
        if not boxes:
            return {'x_min': 0, 'y_min': 0, 'x_max': w, 'y_max': h}
        xs1 = np.asarray([box['x_min'] for box in boxes], dtype=float)
        ys1 = np.asarray([box['y_min'] for box in boxes], dtype=float)
        xs2 = np.asarray([box['x_max'] for box in boxes], dtype=float)
        ys2 = np.asarray([box['y_max'] for box in boxes], dtype=float)
        pad = int(max(metrics['h_line'] * 1.5, min(h, w) * 0.01))
        return {
            'x_min': max(0, int(np.percentile(xs1, 5)) - pad),
            'y_min': max(0, int(np.percentile(ys1, 5)) - pad),
            'x_max': min(w, int(np.percentile(xs2, 95)) + pad),
            'y_max': min(h, int(np.percentile(ys2, 95)) + pad),
        }

    def _discover_structure_separators(self, image: np.ndarray, boxes: List[Dict[str, Any]],
                                       frame: Dict[str, int], metrics: Dict[str, float]) -> Dict[str, List[int]]:
        """Collect rule-line and occupancy-gap evidence in frame coordinates."""
        h, w = image.shape[:2]
        vertical = [x for x in self._detect_vertical_dividers(image)
                    if frame['x_min'] <= x <= frame['x_max']]
        horizontal = [y for y in self._detect_horizontal_dividers(image)
                      if frame['y_min'] <= y <= frame['y_max']]
        # Occupancy gutters complement faint or missing ruled separators.
        vertical.extend(self._find_column_gutters(boxes, w))
        tol = max(2, int(metrics['h_line'] * 0.4))
        def unique(values: List[int]) -> List[int]:
            result: List[int] = []
            for value in sorted(values):
                if not result or value - result[-1] > tol:
                    result.append(int(value))
            return result
        return {'vertical': unique(vertical), 'horizontal': unique(horizontal)}

    def _segment_baselines(self, boxes: List[Dict[str, Any]], frame: Dict[str, int],
                           separators: Dict[str, List[int]], metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Split same-baseline OCR components where a real reading-flow gap exists."""
        if not boxes:
            return []
        baseline_tol = max(3.0, metrics['h_line'] * 0.70)
        # OCR occasionally returns one box spanning a ruled boundary because
        # it saw text from the centre and a neighbouring annotation on the
        # same baseline.  A strong page separator is evidence to split that
        # component into independently analysable crops.  Tiny edge slivers
        # are discarded, so this never fabricates a column from a border.
        frame_width = max(1, frame['x_max'] - frame['x_min'])
        minimum_piece_width = max(metrics['h_line'] * 1.2, frame_width * 0.04)
        pieces: List[Tuple[int, Dict[str, Any]]] = []
        for index, source_box in enumerate(boxes):
            cuts = [source_box['x_min']] + [x for x in separators['vertical']
                    if source_box['x_min'] + minimum_piece_width < x < source_box['x_max'] - minimum_piece_width] + [source_box['x_max']]
            for x1, x2 in zip(cuts, cuts[1:]):
                piece = dict(source_box)
                piece['x_min'], piece['x_max'] = int(x1), int(x2)
                pieces.append((index, piece))

        rows: List[List[Tuple[int, Dict[str, Any]]]] = []
        for index, box in sorted(pieces, key=lambda item: (
            (item[1]['y_min'] + item[1]['y_max']) / 2.0, item[1]['x_min'])):
            center_y = (box['y_min'] + box['y_max']) / 2.0
            best_row = None
            for row in rows:
                med_y = np.median([(item['y_min'] + item['y_max']) / 2.0 for _, item in row])
                if abs(center_y - med_y) <= baseline_tol:
                    # Components in the same horizontal baseline row must not overlap horizontally
                    overlap = False
                    for _, rbox in row:
                        ox = max(0, min(box['x_max'], rbox['x_max']) - max(box['x_min'], rbox['x_min']))
                        min_w = min(box['x_max'] - box['x_min'], rbox['x_max'] - rbox['x_min'])
                        if min_w > 0 and (ox / min_w) > 0.15:
                            overlap = True
                            break
                    if not overlap:
                        best_row = row
                        break
            if best_row is None:
                rows.append([(index, box)])
            else:
                best_row.append((index, box))

        segments: List[Dict[str, Any]] = []
        gap_threshold = max(metrics['h_line'] * 1.8, frame_width * 0.035)
        rule_tol = max(2.0, metrics['h_line'] * 0.6)
        for row in rows:
            ordered = sorted(row, key=lambda item: item[1]['x_min'])
            groups: List[List[Tuple[int, Dict[str, Any]]]] = [[ordered[0]]]
            for item in ordered[1:]:
                previous = groups[-1][-1][1]
                gap = item[1]['x_min'] - previous['x_max']
                has_separator = any(previous['x_max'] - rule_tol <= x <= item[1]['x_min'] + rule_tol
                                    for x in separators['vertical'])
                if gap > gap_threshold or has_separator:
                    groups.append([item])
                else:
                    groups[-1].append(item)
            for group in groups:
                group_boxes = [item[1] for item in group]
                bbox = self._bbox_from_boxes(group_boxes)
                segments.append({
                    'component_indices': [item[0] for item in group],
                    'bbox': bbox,
                    'x_center_norm': ((bbox['x_min'] + bbox['x_max']) / 2.0 - frame['x_min']) / frame_width,
                    'width_norm': (bbox['x_max'] - bbox['x_min']) / frame_width,
                    'y_center_norm': ((bbox['y_min'] + bbox['y_max']) / 2.0 - frame['y_min']) /
                                     max(1, frame['y_max'] - frame['y_min']),
                })
        return sorted(segments, key=lambda segment: (segment['bbox']['y_min'], segment['bbox']['x_min']))

    def _link_reading_flows(self, segments: List[Dict[str, Any]], frame: Dict[str, int],
                            separators: Dict[str, List[int]], metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Link baseline segments using cadence and stable horizontal geometry."""
        if not segments:
            return []
        # A reading flow may bridge ordinary line spacing but not a title/body
        # or body/footer isolation gap.  The robust vertical cadence is much
        # safer than using a large OCR box height, which can include several
        # glyph rows on damaged manuscripts.
        max_gap = max(metrics['g_vert'] * 2.0, metrics['h_line'] * 1.25)
        flows: List[List[int]] = []
        for index, segment in enumerate(segments):
            bbox = segment['bbox']
            best_flow = None
            best_score = -1.0
            for flow in flows:
                previous = segments[flow[-1]]['bbox']
                gap = bbox['y_min'] - previous['y_max']
                if gap < -metrics['h_line'] * 0.5 or gap > max_gap:
                    continue
                if any(previous['y_max'] <= divider <= bbox['y_min']
                       for divider in separators['horizontal']):
                    continue
                overlap = max(0, min(bbox['x_max'], previous['x_max']) - max(bbox['x_min'], previous['x_min']))
                union = max(bbox['x_max'], previous['x_max']) - min(bbox['x_min'], previous['x_min'])
                x_overlap = overlap / max(1, union)
                centre_delta = abs(segment['x_center_norm'] - segments[flow[-1]]['x_center_norm'])
                width_delta = abs(segment['width_norm'] - segments[flow[-1]]['width_norm'])
                previous_is_top_title = (
                    len(flow) == 1 and
                    previous['y_min'] <= frame['y_min'] + 0.12 * max(1, frame['y_max'] - frame['y_min']) and
                    min(segment['width_norm'], segments[flow[-1]]['width_norm']) /
                    max(1e-6, max(segment['width_norm'], segments[flow[-1]]['width_norm'])) < 0.55 and
                    centre_delta <= 0.12
                )
                if previous_is_top_title:
                    continue

                # Isolated narrow text group at the bottom of the frame (attribution, source, or footer)
                # must not be unconditionally fused into an established body flow.
                is_isolated_bottom_line = (
                    len(flow) >= 2 and
                    bbox['y_max'] >= frame['y_max'] - 0.15 * max(1, frame['y_max'] - frame['y_min']) and
                    segment['width_norm'] < (float(np.median([segments[s]['width_norm'] for s in flow]))) * 0.70
                )
                if is_isolated_bottom_line:
                    continue

                score = x_overlap - centre_delta - 0.35 * width_delta
                if (x_overlap >= 0.18 or centre_delta <= 0.10) and score > best_score:
                    best_flow, best_score = flow, score
            if best_flow is None:
                flows.append([index])
            else:
                best_flow.append(index)

        result: List[Dict[str, Any]] = []
        frame_width = max(1, frame['x_max'] - frame['x_min'])
        frame_height = max(1, frame['y_max'] - frame['y_min'])
        for flow_id, segment_ids in enumerate(flows):
            members = [segments[index] for index in segment_ids]
            bbox = self._bbox_from_boxes([member['bbox'] for member in members])
            median_width = float(np.median([member['width_norm'] for member in members]))
            covered_height = (bbox['y_max'] - bbox['y_min']) / frame_height
            nearby_rules = any(bbox['x_min'] <= x <= bbox['x_max'] for x in separators['vertical'])
            result.append({
                'id': flow_id,
                'segment_indices': segment_ids,
                'component_indices': [component for member in members for component in member['component_indices']],
                'bbox': bbox,
                'bbox_norm': {
                    'x_min': (bbox['x_min'] - frame['x_min']) / frame_width,
                    'y_min': (bbox['y_min'] - frame['y_min']) / frame_height,
                    'x_max': (bbox['x_max'] - frame['x_min']) / frame_width,
                    'y_max': (bbox['y_max'] - frame['y_min']) / frame_height,
                },
                'row_count': len(members),
                'median_width_norm': median_width,
                'continuity_score': len(members) * median_width * max(0.05, covered_height),
                'adjacent_separator_evidence': nearby_rules,
            })
        return result

    def reconstruct(
        self,
        image: np.ndarray,
        text_boxes: List[Dict[str, Any]],
        macro_proposals: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct layout regions from fine-grained text boxes and fuse with macro priors.
        Uses normalized geometry and adaptive document metrics.
        """
        h, w = image.shape[:2]
        if not text_boxes:
            logger.debug("[RECONSTRUCT] No text boxes provided; returning macro proposals.")
            return macro_proposals

        # Step 0: Filter ruling-line artifacts and compute adaptive metrics
        valid_boxes = []
        for b in text_boxes:
            bw = b['x_max'] - b['x_min']
            bh = b['y_max'] - b['y_min']
            if bh < 15 and bw / max(1, bh) > 25:
                continue
            if bh < 8 or bw < 8:
                continue
            valid_boxes.append(b)
        if not valid_boxes:
            valid_boxes = list(text_boxes)

        metrics = self.compute_adaptive_metrics(valid_boxes, (h, w))
        h_line = metrics['h_line']
        g_vert = metrics['g_vert']
        w_body = metrics['w_line']

        # Step 1: Detect physical ruling dividers (if present on the page)
        v_dividers = self._detect_vertical_dividers(image)
        top_h_div = self._find_strongest_horizontal_divider(image, int(h * 0.06), int(h * 0.25), exclude_boxes=valid_boxes)
        bottom_h_div = self._find_strongest_horizontal_divider(image, int(h * 0.75), int(h * 0.95), exclude_boxes=valid_boxes)

        # Check macro proposals for layout band corroboration (require confidence >= 0.55)
        macro_footers = [
            m for m in macro_proposals
            if m.get('class') == 'footer' and m.get('confidence', 0) >= 0.55
        ]
        macro_headers = [
            m for m in macro_proposals
            if m.get('class') in {'header', 'doc_title', 'paragraph_title'} and m.get('confidence', 0) >= 0.55
        ]

        sorted_v = sorted(valid_boxes, key=lambda b: (b['y_min'], b['x_min']))

        # Step 1a: Header evidence gate (Specification 4 & Requirement 1)
        header_y_max = 0
        if top_h_div is not None:
            top_text_above_div = [b for b in valid_boxes if b['y_max'] <= top_h_div + 10]
            if top_text_above_div:
                header_y_max = top_h_div
        elif macro_headers:
            header_y_max = max(m['bbox']['y_max'] for m in macro_headers)
        else:
            # Check if line 0 is an isolated top text line (header)
            line0_boxes = [b for b in sorted_v if b['y_min'] <= sorted_v[0]['y_min'] + int(h_line * 0.6)]
            other_boxes = [b for b in sorted_v if b['y_min'] > sorted_v[0]['y_min'] + int(h_line * 0.6)]
            if line0_boxes and other_boxes:
                cand_max_y = max(b['y_max'] for b in line0_boxes)
                next_min_y = min(b['y_min'] for b in other_boxes)
                gap = next_min_y - cand_max_y
                cand_width = max(b['x_max'] for b in line0_boxes) - min(b['x_min'] for b in line0_boxes)
                meaningful_gap = (gap >= max(20.0, 1.5 * g_vert))
                is_distinct = (cand_width <= 0.75 * w_body) or (cand_width <= 0.55 * w)
                if meaningful_gap and is_distinct:
                    header_y_max = cand_max_y + int(gap * 0.5)
                else:
                    header_y_max = 0
            else:
                header_y_max = 0

        # Step 1b: Footer evidence gate (Specification 5 & Requirement 2)
        footer_y_min = h
        if macro_footers:
            footer_y_min = min(m['bbox']['y_min'] for m in macro_footers)
        elif bottom_h_div is not None:
            bot_text_below_div = [b for b in valid_boxes if b['y_min'] >= bottom_h_div - 10]
            if bot_text_below_div:
                footer_y_min = bottom_h_div
        else:
            # Search the lower page for an *isolated* compact text group.  The
            # previous implementation inspected only the absolute last OCR box,
            # so a bottom English attribution could hide a real folio/catchword
            # immediately above it.
            bottom_cut = int(h * 0.93)
            lower = [b for b in sorted_v if b['y_min'] >= int(h * 0.74) and b['y_max'] <= bottom_cut]
            bot_boxes = []
            if lower:
                last = lower[-1]
                bot_boxes = [b for b in lower if abs(((b['y_min'] + b['y_max']) / 2.0) -
                                                     ((last['y_min'] + last['y_max']) / 2.0)) <= max(h_line * 0.75, 6.0)]
            prev_boxes = [b for b in sorted_v if b['y_max'] < (min((x['y_min'] for x in bot_boxes), default=h) - max(1.0, 0.6 * h_line))]
            if bot_boxes and prev_boxes:
                bot_min_y = min(b['y_min'] for b in bot_boxes)
                prev_max_y = max(b['y_max'] for b in prev_boxes)
                gap = bot_min_y - prev_max_y
                cand_width = max(b['x_max'] for b in bot_boxes) - min(b['x_min'] for b in bot_boxes)
                cand_height = max(b['y_max'] for b in bot_boxes) - bot_min_y
                meaningful_gap = (gap >= max(1.5 * h_line, 1.5 * g_vert))
                is_footer_like = (
                    cand_height <= 2.2 * h_line and
                    (cand_width <= 0.65 * w_body or cand_width < 0.50 * w or len(bot_boxes) <= 2)
                )
                if meaningful_gap and is_footer_like:
                    footer_y_min = prev_max_y + int(gap * 0.5)
                else:
                    footer_y_min = h
            else:
                footer_y_min = h

        # Step 1c: Adaptive filler boundary (Specification 6)
        filler_y_min = int(h * 0.93)
        if bottom_h_div is not None and bottom_h_div > footer_y_min:
            filler_y_min = max(filler_y_min, bottom_h_div)

        # Detect column gutters if physical vertical lines are not drawn
        if len(v_dividers) < 2 and len(valid_boxes) >= 6:
            gutter_divs = self._find_column_gutters(valid_boxes, w)
            if len(gutter_divs) >= 2:
                v_dividers = gutter_divs

        col_tol = int(max(8, h_line * 0.6))

        # Step 2: Adaptive vertical band partitioning
        pad_band = int(max(4, h_line * 0.35))
        top_boxes: List[Dict[str, Any]] = []
        bottom_boxes: List[Dict[str, Any]] = []
        mid_boxes: List[Dict[str, Any]] = []

        for b in valid_boxes:
            ymin, ymax = b['y_min'], b['y_max']
            if header_y_max > 0 and ymax <= header_y_max + pad_band:
                top_boxes.append(b)
            elif footer_y_min < h and ymin >= footer_y_min - pad_band:
                # If vertical margin dividers exist, boxes in the left or right margin columns
                # belong to the margin columns (side_text or filler), NOT to the bottom footer!
                if len(v_dividers) >= 2:
                    d_left, d_right = v_dividers[0], v_dividers[-1]
                    if b['x_max'] <= d_left + col_tol or (b['x_min'] + b['x_max']) / 2.0 >= d_right - col_tol:
                        mid_boxes.append(b)
                        continue
                bottom_boxes.append(b)
            else:
                mid_boxes.append(b)

        logger.debug(f"[RECONSTRUCT] Adaptive metrics: h_line={h_line:.1f}, g_vert={g_vert:.1f}")
        logger.debug(f"[RECONSTRUCT] Detected {len(v_dividers)} vertical dividers: {v_dividers}")
        logger.debug(
            f"[RECONSTRUCT] Normalized dividers: top={header_y_max / h:.3f}, footer={footer_y_min / h:.3f}, filler={filler_y_min / h:.3f}"
        )

        reconstructed: List[Dict[str, Any]] = []

        # Step 3: Reconstruct Header (Specification 4)
        if top_boxes:
            h_pad_x = int(max(6, h_line * 0.5))
            h_pad_y = int(max(4, g_vert * 0.5))
            h_box = self._envelope_from_boxes(top_boxes, pad_x=h_pad_x, pad_y=h_pad_y, max_w=w, max_h=h)
            h_score = float(np.mean([b['score'] for b in top_boxes]))
            reconstructed.append({
                'class': 'header',
                'confidence': round(float(np.clip(h_score, 0.05, 0.99)), 4),
                'ocr_det_conf': round(float(np.clip(h_score, 0.05, 0.99)), 4),
                'bbox': h_box,
            })
            logger.debug(f"[RECONSTRUCT] Built header from {len(top_boxes)} text boxes: {h_box}")

        # Step 4: Reconstruct Mid-Body Columns (Specifications 2, 3, and 8)
        col_tol = int(max(8, h_line * 0.6))

        if len(v_dividers) >= 2 and mid_boxes:
            d_left, d_right = v_dividers[0], v_dividers[-1]
            left_col = [b for b in mid_boxes if b['x_max'] <= d_left + col_tol]
            center_col = [
                b for b in mid_boxes
                if b['x_min'] >= d_left - col_tol and b['x_max'] <= d_right + col_tol
            ]
            right_col = [b for b in mid_boxes if b['x_min'] >= d_right - col_tol]

            # Positive evidence check for left margin
            left_has_macro = any(
                m.get('class') == 'aside_text' and m.get('confidence', 0) >= 0.55
                and m['bbox']['x_max'] <= d_left + col_tol
                for m in macro_proposals
            )
            left_area = sum((b['x_max'] - b['x_min']) * (b['y_max'] - b['y_min']) for b in left_col)
            left_valid = (len(left_col) >= 3) or (len(left_col) >= 2 and left_area >= 2000) or (left_has_macro and len(left_col) >= 1)

            # Positive evidence check for right margin
            right_has_macro = any(
                m.get('class') == 'aside_text' and m.get('confidence', 0) >= 0.55
                and m['bbox']['x_min'] >= d_right - col_tol
                for m in macro_proposals
            )
            right_area = sum((b['x_max'] - b['x_min']) * (b['y_max'] - b['y_min']) for b in right_col)
            right_valid = (len(right_col) >= 3) or (len(right_col) >= 2 and right_area >= 2000) or (right_has_macro and len(right_col) >= 1)

            if not left_valid and not right_valid:
                # Page is not a multi-column commentary! Ruling lines were just outer margin borders.
                center_col = list(mid_boxes)
                left_col = []
                right_col = []
            else:
                if not left_valid:
                    center_col.extend(left_col)
                    left_col = []
                if not right_valid:
                    center_col.extend(right_col)
                    right_col = []

            logger.debug(
                f"[RECONSTRUCT] Multi-column split: left={len(left_col)} (valid={left_valid}), "
                f"center={len(center_col)}, right={len(right_col)} (valid={right_valid})"
            )

            # Left side text: tight envelope to content (Specification 3)
            if left_col and left_valid:
                b_left = self._envelope_from_boxes(left_col, pad_x=int(h_line * 0.3), pad_y=int(g_vert * 0.4), max_w=w, max_h=h)
                b_left['x_max'] = min(b_left['x_max'], d_left)
                s_left = float(np.mean([b['score'] for b in left_col]))
                reconstructed.append({
                    'class': 'side_text',
                    'confidence': round(float(np.clip(s_left, 0.05, 0.99)), 4),
                    'ocr_det_conf': round(float(np.clip(s_left, 0.05, 0.99)), 4),
                    'bbox': b_left,
                    'text_boxes': left_col,
                })

            # Center main text (Specification 2: Dominant body column)
            if center_col:
                b_center = self._envelope_from_boxes(center_col, pad_x=int(h_line * 0.3), pad_y=int(g_vert * 0.4), max_w=w, max_h=h)
                if left_valid:
                    b_center['x_min'] = max(b_center['x_min'], d_left)
                if right_valid:
                    b_center['x_max'] = min(b_center['x_max'], d_right)
                s_center = float(np.mean([b['score'] for b in center_col]))
                reconstructed.append({
                    'class': 'main_text',
                    'confidence': round(float(np.clip(s_center, 0.05, 0.99)), 4),
                    'ocr_det_conf': round(float(np.clip(s_center, 0.05, 0.99)), 4),
                    'bbox': b_center,
                })

            # Right side text: tight envelope to content (Specification 3)
            if right_col and right_valid:
                b_right = self._envelope_from_boxes(right_col, pad_x=int(h_line * 0.3), pad_y=int(g_vert * 0.4), max_w=w, max_h=h)
                b_right['x_min'] = max(b_right['x_min'], d_right)
                s_right = float(np.mean([b['score'] for b in right_col]))
                reconstructed.append({
                    'class': 'side_text',
                    'confidence': round(float(np.clip(s_right, 0.05, 0.99)), 4),
                    'ocr_det_conf': round(float(np.clip(s_right, 0.05, 0.99)), 4),
                    'bbox': b_right,
                    'text_boxes': right_col,
                })
        elif mid_boxes:
            # Single-column or multi-block manuscript body (Specification 2 & 8)
            blocks = self._cluster_vertical_blocks(mid_boxes, h_line, g_vert, w, h)
            for blk in blocks:
                b_main = self._envelope_from_boxes(blk, pad_x=int(h_line * 0.4), pad_y=int(g_vert * 0.5), max_w=w, max_h=h)
                s_main = float(np.mean([b['score'] for b in blk]))
                reconstructed.append({
                    'class': 'main_text',
                    'confidence': round(float(np.clip(s_main, 0.05, 0.99)), 4),
                    'ocr_det_conf': round(float(np.clip(s_main, 0.05, 0.99)), 4),
                    'bbox': b_main,
                })
            logger.debug(f"[RECONSTRUCT] Built {len(blocks)} main_text block(s) from {len(mid_boxes)} text boxes")

        # Step 5: Reconstruct Footer & Filler (Specifications 5 and 6)
        if bottom_boxes:
            foot_boxes = [b for b in bottom_boxes if b['y_min'] < filler_y_min]
            fill_boxes = [b for b in bottom_boxes if b['y_min'] >= filler_y_min]

            # Footer: Colophon / catchwords / folios (Specification 5)
            # Support multiple independent footer regions on the same page
            if foot_boxes:
                foot_boxes.sort(key=lambda b: b['x_min'])
                clusters = []
                cur_cl = [foot_boxes[0]]
                for b in foot_boxes[1:]:
                    if b['x_min'] - cur_cl[-1]['x_max'] <= 80:
                        cur_cl.append(b)
                    else:
                        clusters.append(cur_cl)
                        cur_cl = [b]
                clusters.append(cur_cl)

                # Never stop after detecting the first footer; do not convert additional footer-like regions into filler.
                # Build an independent footer detection for every cluster and ensure it stays inside page boundaries.
                for cl_idx, cl in enumerate(clusters):
                    b_foot = self._envelope_from_boxes(cl, pad_x=int(h_line * 0.3), pad_y=int(g_vert * 0.3), max_w=w, max_h=h)
                    if bottom_h_div is not None and bottom_h_div > footer_y_min:
                        b_foot['y_max'] = min(h, bottom_h_div)
                    # Ensure all footer boxes stay inside page boundaries
                    b_foot['x_min'] = max(0, min(w, b_foot['x_min']))
                    b_foot['y_min'] = max(0, min(h, b_foot['y_min']))
                    b_foot['x_max'] = max(0, min(w, b_foot['x_max']))
                    b_foot['y_max'] = max(0, min(h, b_foot['y_max']))
                    if b_foot['x_max'] <= b_foot['x_min'] or b_foot['y_max'] <= b_foot['y_min']:
                        continue

                    s_foot = float(np.mean([b['score'] for b in cl]))
                    reconstructed.append({
                        'class': 'footer',
                        'confidence': round(float(np.clip(s_foot, 0.05, 0.99)), 4),
                        'ocr_det_conf': round(float(np.clip(s_foot, 0.05, 0.99)), 4),
                        'bbox': b_foot,
                        'text_boxes': cl,
                    })
                    logger.debug(f"[RECONSTRUCT] Built footer region {cl_idx + 1} from {len(cl)} text boxes: {b_foot}")

            # Filler: Modern citation / attribution metadata (Specification 6)
            if fill_boxes:
                b_fill = self._envelope_from_boxes(fill_boxes, pad_x=int(h_line * 0.3), pad_y=int(g_vert * 0.3), max_w=w, max_h=h)
                s_fill = float(np.mean([b['score'] for b in fill_boxes]))
                reconstructed.append({
                    'class': 'filler',
                    'confidence': round(float(np.clip(s_fill, 0.05, 0.99)), 4),
                    'ocr_det_conf': round(float(np.clip(s_fill, 0.05, 0.99)), 4),
                    'bbox': b_fill,
                    'text_boxes': fill_boxes,
                })
                logger.debug(f"[RECONSTRUCT] Built filler from {len(fill_boxes)} text boxes: {b_fill}")

        # Step 6: Multi-Signal Decorative Elements from Macro Priors (Specification 6)
        # Seals, figures, decorative vignettes, and charts are mapped directly to filler
        for m in macro_proposals:
            m_cls = m.get('class')
            if m_cls in {'figure', 'seal', 'watermark', 'chart', 'decoration'}:
                mb = m['bbox']
                reconstructed.append({
                    'class': 'filler',
                    'confidence': round(m.get('confidence', 0.85), 4),
                    'bbox': mb,
                })

        # Step 7: Two-Source Fusion and Container Suppression (Specifications 7 and 8)
        fused = self._fuse_with_macro_priors(reconstructed, macro_proposals, image_shape=(h, w))
        return fused

    def _cluster_vertical_blocks(
        self,
        boxes: List[Dict[str, Any]],
        h_line: float,
        g_vert: float,
        w: int,
        h: int,
    ) -> List[List[Dict[str, Any]]]:
        """
        Group text boxes into coherent vertical blocks using adaptive gaps.
        If boxes are separated by a large vertical gap (> 3.5 * g_vert), splits into separate blocks.
        """
        if not boxes:
            return []

        sorted_boxes = sorted(boxes, key=lambda b: (b['y_min'], b['x_min']))
        max_gap = max(g_vert * 3.5, h_line * 2.5)

        blocks: List[List[Dict[str, Any]]] = []
        cur_blk = [sorted_boxes[0]]

        for b in sorted_boxes[1:]:
            prev_b = cur_blk[-1]
            gap = b['y_min'] - prev_b['y_max']
            if gap <= max_gap:
                cur_blk.append(b)
            else:
                blocks.append(cur_blk)
                cur_blk = [b]

        if cur_blk:
            blocks.append(cur_blk)

        filtered_blocks: List[List[Dict[str, Any]]] = []
        for blk in blocks:
            blk_h = max(b['y_max'] for b in blk) - min(b['y_min'] for b in blk)
            blk_w = max(b['x_max'] for b in blk) - min(b['x_min'] for b in blk)
            blk_area = blk_w * blk_h
            if blk_h < 15 or blk_area < 0.002 * (w * h):
                continue
            filtered_blocks.append(blk)

        return filtered_blocks if filtered_blocks else blocks

    def _detect_vertical_dividers(self, image: np.ndarray) -> List[int]:
        """Detect internal vertical dividing lines separating columns using morphological opening."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(h * 0.05))))
        edges = cv2.Canny(gray, 50, 150)
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)

        v_proj = np.sum(v_lines[int(h * 0.20):int(h * 0.80), :], axis=0)
        if v_proj.max() == 0:
            return []

        threshold = np.percentile(v_proj, 97)
        v_peaks = np.where(v_proj > threshold)[0]

        dividers: List[int] = []
        min_gap = int(w * 0.05)
        for p in v_peaks:
            if int(w * 0.08) < p < int(w * 0.92):
                if not dividers or (p - dividers[-1]) > min_gap:
                    dividers.append(int(p))

        return dividers

    def _find_column_gutters(self, boxes: List[Dict[str, Any]], w: int) -> List[int]:
        """Detect column gutters from text box distribution when ruling lines are absent."""
        if not boxes or w <= 0:
            return []
        occ = np.zeros(w, dtype=int)
        for b in boxes:
            x1 = max(0, min(w, b['x_min']))
            x2 = max(0, min(w, b['x_max']))
            if x2 > x1:
                occ[x1:x2] += 1

        sub_start = int(w * 0.12)
        sub_end = int(w * 0.88)
        sub_occ = occ[sub_start:sub_end]

        min_gutter_w = max(10, int(w * 0.015))
        is_gap = sub_occ <= max(0, int(len(boxes) * 0.05))

        gap_runs = []
        in_gap = False
        gap_s = 0
        for i, g in enumerate(is_gap):
            if g and not in_gap:
                in_gap = True
                gap_s = i
            elif not g and in_gap:
                in_gap = False
                if (i - gap_s) >= min_gutter_w:
                    gap_runs.append((sub_start + gap_s, sub_start + i))
        if in_gap and (len(is_gap) - gap_s) >= min_gutter_w:
            gap_runs.append((sub_start + gap_s, sub_start + len(is_gap)))

        gutters = [int((s + e) / 2) for s, e in gap_runs]
        return gutters

    def _detect_horizontal_dividers(self, image: np.ndarray) -> List[int]:
        """Detect horizontal dividing lines separating header/footer bands."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * 0.08)), 1))
        edges = cv2.Canny(gray, 50, 150)
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)

        h_proj = np.sum(h_lines[:, int(w * 0.10):int(w * 0.90)], axis=1)
        if h_proj.max() == 0:
            return []

        threshold = np.percentile(h_proj, 97)
        h_peaks = np.where(h_proj > threshold)[0]

        dividers: List[int] = []
        min_gap = int(h * 0.05)
        for p in h_peaks:
            if int(h * 0.05) < p < int(h * 0.95):
                if not dividers or (p - dividers[-1]) > min_gap:
                    dividers.append(int(p))

        return dividers

    def _find_strongest_horizontal_divider(
        self, image: np.ndarray, y_start: int, y_end: int,
        exclude_boxes: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[int]:
        """Find a genuine drawn horizontal ruling line in the given vertical range."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(w * 0.12)), 1))
        edges = cv2.Canny(gray, 50, 150)
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
        h_proj = np.sum(h_lines[:, int(w * 0.10):int(w * 0.90)], axis=1)

        if h_proj.max() == 0:
            return None

        sub_range = [y for y in range(max(0, int(y_start)), min(h, int(y_end)))]
        if exclude_boxes:
            sub_range = [
                y for y in sub_range
                if not any(b['y_min'] - 4 <= y <= b['y_max'] + 4 for b in exclude_boxes)
            ]
        if not sub_range:
            return None

        best_y = max(sub_range, key=lambda y: h_proj[y])
        min_line_len = (w * 0.80) * 0.25 * 255
        if h_proj[best_y] > np.percentile(h_proj, 90) and h_proj[best_y] >= min_line_len:
            return int(best_y)
        return None

    def _envelope_from_boxes(
        self,
        boxes: List[Dict[str, Any]],
        pad_x: int = 5,
        pad_y: int = 5,
        max_w: int = 4000,
        max_h: int = 4000,
    ) -> Dict[str, int]:
        """Compute padded axis-aligned bounding box enclosing all given text boxes."""
        xmin = max(0, min(b['x_min'] for b in boxes) - pad_x)
        ymin = max(0, min(b['y_min'] for b in boxes) - pad_y)
        xmax = min(max_w, max(b['x_max'] for b in boxes) + pad_x)
        ymax = min(max_h, max(b['y_max'] for b in boxes) + pad_y)
        return {'x_min': xmin, 'y_min': ymin, 'x_max': xmax, 'y_max': ymax}

    def _fuse_with_macro_priors(
        self,
        reconstructed: List[Dict[str, Any]],
        macro_proposals: List[Dict[str, Any]],
        image_shape: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """
        Reconcile reconstructed fine-grained regions with macro PP-DocLayoutV2 proposals.
        Suppresses giant macro container boxes if multiple reconstructed regions exist.
        Never allows main_text to swallow other semantic regions.
        """
        h, w = image_shape
        page_area = float(h * w)

        if not reconstructed:
            return macro_proposals

        final_regions: List[Dict[str, Any]] = []
        has_multi_regions = len(reconstructed) >= 2

        if has_multi_regions:
            final_regions.extend(reconstructed)

            for m in macro_proposals:
                m_cls = m['class']
                mb = m['bbox']
                m_area = (mb['x_max'] - mb['x_min']) * (mb['y_max'] - mb['y_min'])

                # If it's a giant container (> 40% of page), suppress it (Specification 7)
                if m_area / page_area > 0.40:
                    logger.debug(f"[FUSION] Suppressed giant macro container box: {mb}")
                    continue

                overlaps = False
                for r in reconstructed:
                    iou = self._compute_iou(mb, r['bbox'])
                    ix1 = max(mb['x_min'], r['bbox']['x_min'])
                    iy1 = max(mb['y_min'], r['bbox']['y_min'])
                    ix2 = min(mb['x_max'], r['bbox']['x_max'])
                    iy2 = min(mb['y_max'], r['bbox']['y_max'])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    r_area = (r['bbox']['x_max'] - r['bbox']['x_min']) * (r['bbox']['y_max'] - r['bbox']['y_min'])
                    containment = inter / max(1, min(m_area, r_area))

                    if iou > 0.35 or containment > 0.65 or (m_cls == r['class'] and inter > 0):
                        overlaps = True
                        break
                if not overlaps and m_cls in {'filler', 'header', 'footer', 'side_text'}:
                    final_regions.append(m)
        else:
            final_regions.extend(reconstructed)

        return final_regions

    def _compute_iou(self, b1: Dict[str, int], b2: Dict[str, int]) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        ix1 = max(b1['x_min'], b2['x_min'])
        iy1 = max(b1['y_min'], b2['y_min'])
        ix2 = min(b1['x_max'], b2['x_max'])
        iy2 = min(b1['y_max'], b2['y_max'])

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection = float(iw * ih)

        a1 = (b1['x_max'] - b1['x_min']) * (b1['y_max'] - b1['y_min'])
        a2 = (b2['x_max'] - b2['x_min']) * (b2['y_max'] - b2['y_min'])
        union = a1 + a2 - intersection

        return intersection / union if union > 0 else 0.0
