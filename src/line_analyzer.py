"""
Line-level text detection, extraction, and script analysis for manuscript layout refinement.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from src.utils import suppress_c_output

logger = logging.getLogger('manuscript-detector')


def count_script_characters(text: str) -> Tuple[int, int, int]:
    """
    Count Latin alphabetic, Devanagari/Indic, and total non-whitespace characters.
    
    Returns:
        (latin_count, devanagari_count, total_count)
    """
    latin_count = 0
    devanagari_count = 0
    total_count = 0
    
    for c in text:
        if c.isspace():
            continue
        total_count += 1
        
        # Latin
        if c.isascii() and c.isalpha():
            latin_count += 1
        # Devanagari and Vedic extensions
        elif ('\u0900' <= c <= '\u097F') or ('\uA8E0' <= c <= '\uA8FF') or ('\u1CD0' <= c <= '\u1CFF'):
            devanagari_count += 1
        # Other Indic scripts (Grantha, Sharada, Kannada, Telugu, Bengali)
        elif ('\u11300' <= c <= '\u1137F') or ('\u11180' <= c <= '\u111DF') or \
             ('\u0C80' <= c <= '\u0CFF') or ('\u0C00' <= c <= '\u0C7F') or ('\u0980' <= c <= '\u09FF'):
            devanagari_count += 1
            
    return latin_count, devanagari_count, total_count


class LineAnalyzer:
    """
    Extracts individual text lines within an ROI and analyzes their
    geometric, visual, and script characteristics.
    """

    def __init__(self, recognizer=None, config: Optional[dict] = None, profiler: Any = None):
        self.recognizer = recognizer
        self.profiler = profiler
        self.full_config = config or {}
        self.config = self.full_config.get('semantic_refinement', {}) if isinstance(config, dict) else {}
        perf_cfg = self.full_config.get('performance', {}) if isinstance(self.full_config, dict) else {}
        self.fast_mode = perf_cfg.get('fast_mode', self.full_config.get('fast_mode', True) if isinstance(self.full_config, dict) else True)
        self.min_line_h = self.config.get('min_line_height', 4)
        self.pad = self.config.get('padding', 2)

    def extract_tight_line_x_span(self, line_img: np.ndarray,
                                  block_size: int = 10,
                                  min_density: float = 0.08) -> Tuple[int, int]:
        """
        Calculates the tight horizontal extent [x_min, x_max] of text glyphs inside a line strip,
        preventing outer photographic scan borders or blank margins from being absorbed.
        """
        if line_img.size == 0:
            return 0, 0
            
        gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY) if len(line_img.shape) == 3 else line_img
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        thr = max(10, int(np.mean(grad) + 0.25 * np.std(grad)))
        fg = grad > thr
        rh, rw = fg.shape
        if rw == 0:
            return 0, 0

        n_blocks = (rw + block_size - 1) // block_size
        densities = [np.mean(fg[:, i * block_size : min(rw, (i + 1) * block_size)]) for i in range(n_blocks)]
        active = [d >= min_density for d in densities]

        # Bridge small inter-word spaces (1 to 2 blocks)
        bridged = list(active)
        for i in range(1, len(bridged) - 1):
            if not bridged[i] and bridged[i - 1] and bridged[i + 1]:
                bridged[i] = True
        for i in range(1, len(bridged) - 2):
            if not bridged[i] and not bridged[i + 1] and bridged[i - 1] and bridged[i + 2]:
                bridged[i] = True
                bridged[i + 1] = True

        runs = []
        cur = []
        for idx, act in enumerate(bridged):
            if act:
                cur.append(idx)
            else:
                if cur:
                    runs.append(cur)
                    cur = []
        if cur:
            runs.append(cur)

        if not runs:
            return 0, rw

        best_run = max(runs, key=len)
        start_x = best_run[0] * block_size
        end_x = min(rw, (best_run[-1] + 1) * block_size)

        sub_fg = fg[:, start_x:end_x]
        ink_cols = np.where(np.sum(sub_fg, axis=0) >= 1)[0]
        if len(ink_cols) > 0:
            start_x = start_x + int(ink_cols[0])
            end_x = start_x + int(ink_cols[-1] - ink_cols[0]) + 1

        return int(start_x), int(end_x)

    def extract_lines(self, image: np.ndarray, bbox: Dict[str, int],
                      max_ocr_lines: Optional[int] = None,
                      sample_only: bool = False) -> List[Dict[str, Any]]:
        """
        Extract line-level bounding boxes and metadata inside a candidate ROI.
        
        Uses gradient-based horizontal projection to find line bands,
        then sub-segments oversized bands to produce proper single-line strips
        for reliable OCR recognition.
        
        Args:
            image: Full original BGR image.
            bbox: Candidate region bounding box {'x_min', 'y_min', 'x_max', 'y_max'}.
            max_ocr_lines: Maximum number of lines to OCR in this region (Fast Mode).
            sample_only: If True, only OCR the first line to confirm script (e.g. side_text).
            
        Returns:
            List of line dictionaries with geometry, OCR, and script metrics.
        """
        h, w = image.shape[:2]
        x1 = max(0, int(bbox['x_min']))
        y1 = max(0, int(bbox['y_min']))
        x2 = min(w, int(bbox['x_max']))
        y2 = min(h, int(bbox['y_max']))

        roi = image[y1:y2, x1:x2]
        if roi.size == 0 or (y2 - y1) < self.min_line_h:
            return []

        box_h = y2 - y1
        box_w = x2 - x1
        max_line_h = self.config.get('max_single_line_height', 35)

        # Case A: If candidate box is a single line (<= max_line_h height)
        if box_h <= max_line_h:
            line_dict = self._analyze_single_line_crop(image, x1, y1, x2, y2, 0, 0, box_h)
            return [line_dict] if line_dict else []

        # Case B: Multi-line candidate box. Find internal horizontal line bands.
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        thr = max(10, int(np.mean(grad) + 0.25 * np.std(grad)))
        fg = grad > thr

        # Suppress vertical border/ruler lines so horizontal projection isolates text rows
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(box_h * 0.15))))
        vert_lines = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, vert_kernel)
        text_fg = (fg & (~(vert_lines > 0)))

        h_proj = np.sum(text_fg, axis=1)
        peak = np.max(h_proj) if len(h_proj) > 0 else 0
        adaptive_thr = max(10, min(int(box_w * 0.05), int(0.12 * peak)))
        is_line = h_proj >= adaptive_thr

        bands = []
        in_band = False
        start_y = 0
        for y, act in enumerate(is_line):
            if act and not in_band:
                in_band = True
                start_y = y
            elif not act and in_band:
                in_band = False
                if y - start_y >= self.min_line_h:
                    bands.append((start_y, y))
        if in_band and (len(is_line) - start_y) >= self.min_line_h:
            bands.append((start_y, len(is_line)))

        if not bands:
            single = self._analyze_single_line_crop(image, x1, y1, x2, y2, 0, 0, box_h)
            return [single] if single else []

        # Sub-segment oversized bands into individual text lines
        final_bands = []
        for (by0, by1) in bands:
            band_h = by1 - by0
            if band_h > max_line_h * 1.5:
                # This band is too tall to be a single line — sub-segment it
                sub_bands = self._subsegment_band(text_fg[by0:by1, :], by0, band_h, max_line_h)
                if sub_bands:
                    final_bands.extend(sub_bands)
                else:
                    final_bands.append((by0, by1))
            else:
                final_bands.append((by0, by1))

        lines = []
        n_bands = len(final_bands)

        band_items = []
        for i, (by0, by1) in enumerate(final_bands):
            abs_y0 = y1 + by0
            abs_y1 = y1 + by1
            gap_above = by0 if i == 0 else by0 - final_bands[i - 1][1]
            gap_below = (box_h - by1) if i == n_bands - 1 else final_bands[i + 1][0] - by1

            line_crop = image[abs_y0:abs_y1, x1:x2]
            if line_crop.size == 0:
                continue

            line_h = abs_y1 - abs_y0
            lx1_rel, lx2_rel = self.extract_tight_line_x_span(line_crop)
            if lx2_rel > lx1_rel:
                tight_x1 = max(0, x1 + lx1_rel - self.pad)
                tight_x2 = min(w, x1 + lx2_rel + self.pad)
            else:
                tight_x1, tight_x2 = x1, x2

            tight_y1 = max(0, abs_y0 - self.pad)
            tight_y2 = min(h, abs_y1 + self.pad)
            line_w = tight_x2 - tight_x1
            y_center = (abs_y0 + abs_y1) / 2.0
            rel_pos = y_center / max(1, h)

            band_items.append({
                'abs_y0': abs_y0,
                'abs_y1': abs_y1,
                'tight_x1': tight_x1,
                'tight_y1': tight_y1,
                'tight_x2': tight_x2,
                'tight_y2': tight_y2,
                'line_h': line_h,
                'line_w': line_w,
                'rel_pos': rel_pos,
                'gap_above': gap_above,
                'gap_below': gap_below,
                'band_idx': i,
            })

        # Identify candidate lines for OCR: peripheral lines (first 2 / last 2),
        # lines with an isolation gap, or lines near outer page margins.
        crops_to_ocr = []
        ocr_indices = []
        max_ocr_h = self.config.get('max_single_line_height', 35)

        # Filter candidate bands: prioritize real text lines over border noise (line_h >= 10)
        valid_bands = [item for item in band_items if item['line_h'] >= 10]
        if not valid_bands:
            valid_bands = band_items

        if sample_only and valid_bands:
            # Fast script validation: pick the tallest/most salient valid line
            item = max(valid_bands, key=lambda it: it['line_h'])
            if self.recognizer is not None:
                rec_crop = image[item['abs_y0']:item['abs_y1'], item['tight_x1']:item['tight_x2']]
                if rec_crop.size > 0 and rec_crop.shape[1] > 10:
                    crops_to_ocr.append(rec_crop)
                    ocr_indices.append(item['band_idx'])
        elif self.fast_mode:
            # FAST MODE: Never OCR every manuscript line.
            # In multi-line manuscript blocks, OCR ONLY peripheral boundary lines
            # (bottom attribution or top margin notes) or lines with large isolation gaps.
            candidate_indices = []
            if len(valid_bands) <= 2:
                candidate_indices = [it['band_idx'] for it in valid_bands]
            else:
                # Prioritize bottom valid line (where modern attributions / citations occur)
                candidate_indices.append(valid_bands[-1]['band_idx'])
                # Prioritize top valid line (where headers / cataloguer notes occur)
                if len(valid_bands) > 1 and valid_bands[0]['band_idx'] not in candidate_indices:
                    candidate_indices.append(valid_bands[0]['band_idx'])
                # Check any internal line with large isolation gaps
                for it in valid_bands[1:-1]:
                    if it['gap_above'] >= 25 or it['gap_below'] >= 25:
                        candidate_indices.append(it['band_idx'])

            limit = max_ocr_lines if max_ocr_lines is not None else 3
            candidate_indices = candidate_indices[:limit]

            for idx in candidate_indices:
                item = next((it for it in band_items if it['band_idx'] == idx), None)
                if item and self.recognizer is not None:
                    rec_crop = image[item['abs_y0']:item['abs_y1'], item['tight_x1']:item['tight_x2']]
                    if rec_crop.size > 0 and rec_crop.shape[1] > 10:
                        crops_to_ocr.append(rec_crop)
                        ocr_indices.append(item['band_idx'])
        else:
            for item in band_items:
                i = item['band_idx']
                needs_ocr = (
                    n_bands <= 4
                    or i < 2
                    or i >= n_bands - 2
                    or item['gap_above'] >= 8
                    or item['gap_below'] >= 8
                    or item['rel_pos'] < 0.18
                    or item['rel_pos'] > 0.82
                )
                if needs_ocr and self.recognizer is not None and item['line_h'] <= max_ocr_h * 1.5:
                    rec_crop = image[item['abs_y0']:item['abs_y1'], item['tight_x1']:item['tight_x2']]
                    if rec_crop.size > 0 and rec_crop.shape[1] > 10:
                        crops_to_ocr.append(rec_crop)
                        ocr_indices.append(i)

        ocr_results_map = {}
        if crops_to_ocr:
            ocr_outputs = self._batch_ocr(crops_to_ocr)
            for idx, res in zip(ocr_indices, ocr_outputs):
                ocr_results_map[idx] = res

        for item in band_items:
            i = item['band_idx']
            ocr_text, ocr_conf = ocr_results_map.get(i, ('', 0.0))
            n_latin, n_deva, n_total = count_script_characters(ocr_text)
            latin_ratio = (n_latin / n_total) if n_total > 0 else 0.0
            deva_ratio = (n_deva / n_total) if n_total > 0 else 0.0

            lines.append({
                'bbox': {
                    'x_min': item['tight_x1'],
                    'y_min': item['tight_y1'],
                    'x_max': item['tight_x2'],
                    'y_max': item['tight_y2'],
                },
                'raw_y_min': item['abs_y0'],
                'raw_y_max': item['abs_y1'],
                'height': item['line_h'],
                'width': item['line_w'],
                'rel_pos': item['rel_pos'],
                'gap_above': item['gap_above'],
                'gap_below': item['gap_below'],
                'ocr_text': ocr_text,
                'ocr_conf': ocr_conf,
                'latin_count': n_latin,
                'devanagari_count': n_deva,
                'total_chars': n_total,
                'latin_ratio': latin_ratio,
                'deva_ratio': deva_ratio,
            })

        return lines

    def _subsegment_band(self, band_fg: np.ndarray, band_offset: int,
                         band_h: int, target_line_h: int) -> List[Tuple[int, int]]:
        """
        Sub-segment an oversized horizontal band into individual text lines
        using dilated projection analysis to find inter-line gaps.
        
        For dense manuscripts where the initial projection merges many lines,
        this uses a more sensitive gap-finding approach with horizontal dilation
        to bridge character gaps while preserving inter-line whitespace.
        """
        if band_h < 10:
            return []

        bh, bw = band_fg.shape

        # Use horizontal dilation to bridge intra-line character gaps,
        # making inter-line gaps more prominent
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, bw // 8), 1))
        dilated = cv2.dilate(band_fg.astype(np.uint8), h_kernel, iterations=1)

        h_proj = np.sum(dilated, axis=1).astype(float)
        if np.max(h_proj) == 0:
            return []

        # Smooth the projection to reduce noise
        kernel_size = max(3, band_h // 20) | 1  # ensure odd
        if kernel_size < len(h_proj):
            h_proj_smooth = np.convolve(h_proj, np.ones(kernel_size) / kernel_size, mode='same')
        else:
            h_proj_smooth = h_proj

        # Find local minima in the smoothed projection — these are inter-line gaps
        peak_val = np.max(h_proj_smooth)
        gap_thr = peak_val * 0.25  # A row with <25% of peak density is a gap

        # Estimate expected number of lines
        est_n_lines = max(2, band_h // max(12, target_line_h))

        # Find gap candidates (rows below gap threshold)
        gap_mask = h_proj_smooth < gap_thr

        # Find contiguous gap regions
        gap_runs = []
        in_gap = False
        gap_start = 0
        for y in range(len(gap_mask)):
            if gap_mask[y] and not in_gap:
                in_gap = True
                gap_start = y
            elif not gap_mask[y] and in_gap:
                in_gap = False
                gap_runs.append((gap_start, y))
        if in_gap:
            gap_runs.append((gap_start, len(gap_mask)))

        # Filter gaps: must be at least 2px wide, not at the very edges
        min_gap_w = 2
        valid_gaps = [
            (gs, ge) for gs, ge in gap_runs
            if (ge - gs) >= min_gap_w and gs > target_line_h * 0.3 and ge < bh - target_line_h * 0.3
        ]

        if not valid_gaps:
            step = max(15, min(target_line_h, band_h // est_n_lines))
            sub_bands = []
            for y in range(0, band_h, step):
                y_end = min(band_h, y + step)
                if y_end - y >= self.min_line_h:
                    sub_bands.append((band_offset + y, band_offset + y_end))
            return sub_bands if len(sub_bands) >= 2 else []

        # Build sub-bands from gap positions
        sub_bands = []
        prev_end = 0
        for gs, ge in valid_gaps:
            gap_mid = (gs + ge) // 2
            if gap_mid - prev_end >= self.min_line_h:
                sub_bands.append((band_offset + prev_end, band_offset + gap_mid))
            prev_end = gap_mid

        # Final band after last gap
        if bh - prev_end >= self.min_line_h:
            sub_bands.append((band_offset + prev_end, band_offset + bh))

        # Only return sub-bands if we actually split into multiple parts
        if len(sub_bands) >= 2:
            return sub_bands
        return []

    def _batch_ocr(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Run batched OCR recognition over a list of image strips."""
        if not crops or self.recognizer is None:
            return [('', 0.0)] * len(crops)
        try:
            batch_sz = min(8, len(crops))
            if hasattr(self, 'profiler') and self.profiler is not None:
                t0 = time.perf_counter()
                with suppress_c_output():
                    res = self.recognizer.predict(crops, batch_size=batch_sz)
                elapsed = time.perf_counter() - t0
                self.profiler.record_ocr(crops, elapsed, batch_size=batch_sz)
            else:
                with suppress_c_output():
                    res = self.recognizer.predict(crops, batch_size=batch_sz)
            out = []
            for r in res:
                if isinstance(r, dict):
                    out.append((r.get('rec_text', ''), float(r.get('rec_score', 0.0))))
                elif isinstance(r, (list, tuple)) and len(r) > 0 and isinstance(r[0], dict):
                    out.append((r[0].get('rec_text', ''), float(r[0].get('rec_score', 0.0))))
                else:
                    out.append(('', 0.0))
            return out
        except TypeError:
            # Fallback for mock objects in tests that only accept single crop
            out = []
            for c in crops:
                try:
                    if hasattr(self, 'profiler') and self.profiler is not None:
                        t0 = time.perf_counter()
                        with suppress_c_output():
                            r = self.recognizer.predict(c)
                        elapsed = time.perf_counter() - t0
                        self.profiler.record_ocr([c], elapsed, batch_size=1)
                    else:
                        with suppress_c_output():
                            r = self.recognizer.predict(c)
                    if r and len(r) > 0 and isinstance(r[0], dict):
                        out.append((r[0].get('rec_text', ''), float(r[0].get('rec_score', 0.0))))
                    else:
                        out.append(('', 0.0))
                except Exception:
                    out.append(('', 0.0))
            return out
        except Exception as e:
            logger.debug(f"Batch OCR execution error: {e}")
            return [('', 0.0)] * len(crops)

    def _analyze_single_line_crop(self, image: np.ndarray,
                                  x1: int, y1: int, x2: int, y2: int,
                                  gap_above: int, gap_below: int,
                                  roi_total_h: int) -> Optional[Dict[str, Any]]:
        """Analyze a single line strip and compute all script and geometric signals."""
        h, w = image.shape[:2]
        line_crop = image[y1:y2, x1:x2]
        if line_crop.size == 0:
            return None

        line_h = y2 - y1

        lx1_rel, lx2_rel = self.extract_tight_line_x_span(line_crop)
        if lx2_rel > lx1_rel:
            tight_x1 = max(0, x1 + lx1_rel - self.pad)
            tight_x2 = min(w, x1 + lx2_rel + self.pad)
        else:
            tight_x1, tight_x2 = x1, x2

        tight_y1 = max(0, y1 - self.pad)
        tight_y2 = min(h, y2 + self.pad)

        ocr_text = ''
        ocr_conf = 0.0
        max_ocr_h = self.config.get('max_single_line_height', 35)

        if self.recognizer is not None:
            try:
                if line_h <= max_ocr_h * 1.5:
                    rec_crop = image[y1:y2, tight_x1:tight_x2]
                    if rec_crop.size > 0 and rec_crop.shape[1] > 10:
                        ocr_res = self._batch_ocr([rec_crop])
                        if ocr_res:
                            ocr_text, ocr_conf = ocr_res[0]
                else:
                    edge_h = min(max_ocr_h, line_h // 3)
                    edge_crops = []
                    for edge_y1, edge_y2 in [(y1, y1 + edge_h), (y2 - edge_h, y2)]:
                        c = image[edge_y1:edge_y2, tight_x1:tight_x2]
                        if c.size > 0 and c.shape[1] > 10:
                            edge_crops.append(c)
                    if edge_crops:
                        res_list = self._batch_ocr(edge_crops)
                        for t, score in res_list:
                            if score > ocr_conf:
                                ocr_text, ocr_conf = t, score
            except Exception as e:
                logger.debug(f"OCR recognition error on line [{tight_x1}, {y1}, {tight_x2}, {y2}]: {e}")

        # Script character counting
        n_latin, n_deva, n_total = count_script_characters(ocr_text)
        latin_ratio = (n_latin / n_total) if n_total > 0 else 0.0
        deva_ratio = (n_deva / n_total) if n_total > 0 else 0.0

        line_w = tight_x2 - tight_x1
        y_center = (y1 + y2) / 2.0
        rel_pos = y_center / max(1, h)

        return {
            'bbox': {
                'x_min': tight_x1,
                'y_min': tight_y1,
                'x_max': tight_x2,
                'y_max': tight_y2,
            },
            'raw_y_min': y1,
            'raw_y_max': y2,
            'height': line_h,
            'width': line_w,
            'rel_pos': rel_pos,
            'gap_above': gap_above,
            'gap_below': gap_below,
            'ocr_text': ocr_text,
            'ocr_conf': ocr_conf,
            'latin_count': n_latin,
            'devanagari_count': n_deva,
            'total_chars': n_total,
            'latin_ratio': latin_ratio,
            'deva_ratio': deva_ratio,
        }

    def analyze_region_ocr(self, image: np.ndarray, bbox: Dict[str, int],
                           pre_extracted_lines: Optional[List[Dict[str, Any]]] = None,
                           sample_only: bool = False) -> Dict[str, Any]:
        """
        Extract lines and compute aggregate semantic OCR metrics for an entire region:
        - recognized text
        - recognition confidence
        - Indic/Devanagari ratio
        - Latin/English ratio
        
        Guarantees zero crashes on degraded or empty image crops.
        """
        if pre_extracted_lines is not None and len(pre_extracted_lines) > 0:
            lines = pre_extracted_lines
        else:
            try:
                lines = self.extract_lines(image, bbox, sample_only=sample_only)
            except Exception as e:
                logger.debug(f"Exception during line extraction in analyze_region_ocr: {e}")
                lines = []

        if not lines:
            try:
                single = self._analyze_single_line_crop(
                    image, bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max'], 0, 0, bbox['y_max'] - bbox['y_min']
                )
                if single:
                    lines = [single]
            except Exception as e:
                logger.debug(f"Exception during single line crop in analyze_region_ocr: {e}")

        if not lines:
            return {
                'ocr_text': '',
                'ocr_conf': 0.0,
                'deva_ratio': 0.0,
                'latin_ratio': 0.0,
                'total_chars': 0,
                'latin_count': 0,
                'devanagari_count': 0,
            }

        text_lines = [l['ocr_text'].strip() for l in lines if l.get('ocr_text', '').strip()]
        combined_text = " ".join(text_lines)
        conf_scores = [l['ocr_conf'] for l in lines if l.get('ocr_conf', 0.0) > 0.0]
        mean_conf = float(np.mean(conf_scores)) if conf_scores else 0.0

        n_latin, n_deva, n_total = count_script_characters(combined_text)
        latin_ratio = (n_latin / n_total) if n_total > 0 else 0.0
        deva_ratio = (n_deva / n_total) if n_total > 0 else 0.0

        return {
            'ocr_text': combined_text,
            'ocr_conf': round(mean_conf, 4),
            'deva_ratio': round(deva_ratio, 4),
            'latin_ratio': round(latin_ratio, 4),
            'total_chars': n_total,
            'latin_count': n_latin,
            'devanagari_count': n_deva,
        }

