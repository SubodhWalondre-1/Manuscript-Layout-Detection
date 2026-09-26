"""
Semantic Refinement Engine for Manuscript Layout Region Detection.

Implements second-stage content analysis inside candidate layout regions:
1. Line-level extraction via LineAnalyzer.
2. Multi-signal semantic scoring (script, attribution keywords, line isolation, position).
3. Excludes filler lines and recomputes main_text geometry ONLY from manuscript lines.
4. Generates tight filler bounding boxes without background or binding holes.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from src.line_analyzer import LineAnalyzer

logger = logging.getLogger('manuscript-detector')

DEFAULT_ATTRIBUTION_KEYWORDS = [
    'public domain', 'digitization', 'digitized', 'funding', 'midf',
    'collection', 'manuscript collection', 'rare book', 'rare book and manuscript',
    'library', 'source', 'copyright', 'archive', 'attribution',
    'cc-0', 'cc0', 'egangotri', 'gangotri', 'sringeri', 'campus', 'watermark',
    'all rights', 'reserved', 'scanned', 'university', 'institute',
    'curated', 'folio', 'press', 'edition', 'publication',
    'note', 'annotat', 'society', 'bengal', 'benares', 'acc', 'date',
    'treatise', 'century', 'handwriting', 'part of', 'main text', 'possible', 'later', 'fragment', 'english',
    'bhandarkar', 'oriental', 'research', 'pune', 'poona', 'mumbai', 'delhi', 'calcutta', 'kolkata',
    'british', 'museum', 'bodleian', 'harvard', 'oxford', 'cambridge', 'csu', 'rare', 'page', 'vol', 'no.', 'copy', 'beng region',
    'coledion', 'govt', 'government', 'repository', 'preservation', 'heritage', 'sponsored', 'endowment'
]

COMMON_ENGLISH_WORDS = {
    'the', 'by', 'of', 'in', 'and', 'for', 'with', 'from', 'at', 'to', 'on', 'is', 'as',
    'press', 'ms', 'no', 'vol', 'folio', 'copy', 'page', 'author', 'title', 'ed',
    'dept', 'centre', 'center', 'digitized', 'digitization', 'collection', 'library',
    'manuscript', 'rare', 'society', 'bengal', 'pune', 'poona', 'mumbai', 'delhi',
    'oxford', 'cambridge', 'harvard', 'british', 'museum', 'transcribed', 'printed',
    'published', 'edition', 'university', 'research', 'institute'
}


class SemanticRefiner:
    """
    Performs second-stage semantic refinement inside candidate layout regions.
    """

    def __init__(self, recognizer=None, config: Optional[dict] = None):
        self.config = (config or {}).get('semantic_refinement', {})
        self.full_config = config or {}
        self.line_analyzer = LineAnalyzer(recognizer=recognizer, config=config)
        
        self.attr_keywords = [
            kw.lower() for kw in self.config.get('attribution_keywords', DEFAULT_ATTRIBUTION_KEYWORDS)
        ]
        self.score_threshold = self.config.get('filler_score_threshold', 0.45)
        self.isolation_gap_threshold = self.config.get('isolation_gap_threshold', 25)
        self.latin_ratio_threshold = self.config.get('latin_ratio_threshold', 0.35)
        self.min_latin_chars = self.config.get('min_latin_chars', 4)
        self.owned_region_padding_ratio = self.config.get('owned_region_padding_ratio', 0.15)
        self.body_flow_score_ratio = self.config.get('body_flow_score_ratio', 0.40)
        self.minimum_body_flow_lines = self.config.get('minimum_body_flow_lines', 2)

        # Page-adaptive role assignment. These are ratios/relationship gates,
        # not absolute image coordinates, so the same logic works across
        # portrait, landscape, palm-leaf and manuscript scans of very different size.
        role_cfg = (config or {}).get('role_assignment', {}) if isinstance(config, dict) else {}
        self.side_width_ratio = float(role_cfg.get('side_width_ratio', 0.68))
        self.header_max_lines = int(role_cfg.get('header_max_lines', 3))
        self.footer_max_lines = int(role_cfg.get('footer_max_lines', 3))
        self.role_gap_line_ratio = float(role_cfg.get('role_gap_line_ratio', 1.20))
        self.body_width_min_ratio = float(role_cfg.get('body_width_min_ratio', 0.55))
        self.header_prefix_max_lines = int(role_cfg.get('header_prefix_max_lines', 3))
        self.header_prefix_gap_ratio = float(role_cfg.get('header_prefix_gap_ratio', 1.35))
        self.macro_header_overlap_ratio = float(role_cfg.get('macro_header_overlap_ratio', 0.35))
        self.footer_max_height_ratio = float(role_cfg.get('footer_max_height_ratio', 0.08))
        self.footer_max_body_overlap_ratio = float(role_cfg.get('footer_max_body_overlap_ratio', 0.12))
        self.footer_min_gap_lines = float(role_cfg.get('footer_min_gap_lines', 1.15))
        self.macro_body_fallback_min_height_ratio = float(role_cfg.get('macro_body_fallback_min_height_ratio', 0.22))
        self.macro_body_fallback_min_ink_density = float(role_cfg.get('macro_body_fallback_min_ink_density', 0.015))
        self._profiler = None

    @property
    def profiler(self):
        return getattr(self, '_profiler', None)

    @profiler.setter
    def profiler(self, value):
        self._profiler = value
        if hasattr(self, 'line_analyzer') and self.line_analyzer is not None:
            self.line_analyzer.profiler = value

    def _english_evidence(self, line: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Return semantic English/Latin evidence without trusting OCR charset alone.

        A Devanagari recognizer can misread Sanskrit glyphs as ASCII gibberish
        (for example ``MATEA.MNeTT``).  ASCII character ratio alone is therefore
        not sufficient to call something filler.  We require lexical evidence
        or several coherent English tokens.
        """
        text = (line.get('ocr_text') or '').strip().lower()
        tokens = re.findall(r'[a-z]{2,}', text)
        common = [t for t in tokens if t in COMMON_ENGLISH_WORDS]
        keywords = [kw for kw in self.attr_keywords if kw in text]

        latin_count = int(line.get('latin_count', 0) or 0)
        latin_ratio = float(line.get('latin_ratio', 0.0) or 0.0)
        deva_ratio = float(line.get('deva_ratio', 0.0) or 0.0)
        ocr_conf = float(line.get('ocr_conf', 0.0) or 0.0)

        lexical_ratio = len(common) / max(1, len(tokens))
        # Strongly coherent English:
        coherent = (
            len(common) >= 2 and
            latin_count >= self.min_latin_chars and
            latin_ratio >= 0.45
        )
        # Strong attribution / archive language:
        keyword_hit = bool(keywords)
        # Generic Latin text is accepted only when OCR is reasonably confident
        # and there are multiple word-like tokens; this blocks Sanskrit->Latin garbage.
        confident_latin = (
            latin_ratio >= 0.82 and
            latin_count >= max(8, self.min_latin_chars * 2) and
            len(tokens) >= 3 and
            ocr_conf >= 0.55 and
            lexical_ratio >= 0.15
        )
        positive = bool(keyword_hit or coherent or confident_latin)

        # Devanagari dominance is an explicit counter-signal.
        if deva_ratio >= 0.35 and not keyword_hit:
            positive = False

        return positive, {
            'keywords': keywords,
            'tokens': tokens,
            'common_words': common,
            'lexical_ratio': lexical_ratio,
            'latin_ratio': latin_ratio,
            'deva_ratio': deva_ratio,
            'ocr_conf': ocr_conf,
        }

    def calculate_filler_score(self, line: Dict[str, Any]) -> float:
        """Score non-manuscript filler using semantic + geometric evidence.

        The important invariant is: *location alone never makes text filler*.
        In particular, normal Sanskrit/Devanagari text near the top/bottom of
        a page remains manuscript content unless there is positive English /
        attribution / decorative evidence.
        """
        positive_english, ev = self._english_evidence(line)
        if ev['keywords']:
            return 0.98
        if positive_english:
            score = 0.78
            if len(ev['common_words']) >= 3:
                score += 0.10
            if ev['ocr_conf'] >= 0.75:
                score += 0.07
            return min(0.98, score)

        # Explicit non-text macro prior (set upstream only for actual visual
        # classes such as image/seal/chart/formula).  This is intentionally
        # separate from a macro *text* prediction so a mistaken broad box
        # cannot turn manuscript lines into filler.
        if line.get('visual_filler_prior') is True:
            return 0.85

        # Small isolated symbols can be decorative filler (e.g. asterisks).
        text = (line.get('ocr_text') or '').strip()
        compact = ''.join(text.split())
        width_norm = float(line.get('features', {}).get('width_norm', 1.0) or 1.0)
        if compact and len(compact) <= 4 and not any(ch.isalnum() for ch in compact) and width_norm <= 0.10:
            return 0.80

        return 0.0

    # Page-level ownership -------------------------------------------------
    # The legacy refine_detections entry point below is retained for callers
    # without full-page OCR components.  New detector runs use these methods
    # so a coarse model proposal never becomes a final semantic region.

    def build_page_inventory(self, image: np.ndarray,
                             text_boxes: List[Dict[str, Any]],
                             structure: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Analyse baseline segments without merging separate same-row columns."""
        h, w = image.shape[:2]
        boxes = [b for b in text_boxes if b.get('x_max', 0) > b.get('x_min', 0)
                 and b.get('y_max', 0) > b.get('y_min', 0)]
        if not boxes:
            return []
        # Page-structure analysis already split each baseline at genuine
        # gutters/rules.  Reuse those segments so left notes and right notes do
        # not become a single, page-wide OCR line with the central body.
        structure_segments = (structure or {}).get('baseline_segments', [])
        if structure_segments:
            lines = []
            segment_flow_ids = {}
            for flow in (structure or {}).get('flows', []):
                for segment_index in flow.get('segment_indices', []):
                    segment_flow_ids[segment_index] = flow['id']
            for segment_index, segment in enumerate(structure_segments):
                component_indices = segment.get('component_indices', [])
                row = [boxes[index] for index in component_indices if 0 <= index < len(boxes)]
                if not row:
                    continue
                bbox = segment['bbox']
                analysed = self.line_analyzer._analyze_single_line_crop(
                    image, bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max'],
                    0, 0, bbox['y_max'] - bbox['y_min']
                )
                line = analysed or {'bbox': bbox, 'ocr_text': '', 'ocr_conf': 0.0,
                                    'latin_count': 0, 'devanagari_count': 0, 'total_chars': 0,
                                    'latin_ratio': 0.0, 'deva_ratio': 0.0}
                line['bbox'] = bbox
                line['components'] = row
                line['height'] = bbox['y_max'] - bbox['y_min']
                line['width'] = bbox['x_max'] - bbox['x_min']
                line['raw_y_min'] = bbox['y_min']
                line['raw_y_max'] = bbox['y_max']
                line['flow_id'] = segment_flow_ids.get(segment_index, f'segment-{segment_index}')
                lines.append(line)
            return sorted(lines, key=lambda line: (line['bbox']['y_min'], line['bbox']['x_min']))

        median_h = float(np.median([b['y_max'] - b['y_min'] for b in boxes]))
        baseline_tol = max(3.0, 0.70 * median_h)
        rows: List[List[Dict[str, Any]]] = []
        for box in sorted(boxes, key=lambda b: ((b['y_min'] + b['y_max']) / 2.0, b['x_min'])):
            center_y = (box['y_min'] + box['y_max']) / 2.0
            target = None
            for row in rows:
                med_y = np.median([(item['y_min'] + item['y_max']) / 2.0 for item in row])
                if abs(center_y - med_y) <= baseline_tol:
                    overlap = False
                    for rbox in row:
                        ox = max(0, min(box['x_max'], rbox['x_max']) - max(box['x_min'], rbox['x_min']))
                        min_w = min(box['x_max'] - box['x_min'], rbox['x_max'] - rbox['x_min'])
                        if min_w > 0 and (ox / min_w) > 0.15:
                            overlap = True
                            break
                    if not overlap:
                        target = row
                        break
            if target is None:
                rows.append([box])
            else:
                target.append(box)

        lines = []
        for row in rows:
            bbox = {
                'x_min': max(0, int(min(b['x_min'] for b in row))),
                'y_min': max(0, int(min(b['y_min'] for b in row))),
                'x_max': min(w, int(max(b['x_max'] for b in row))),
                'y_max': min(h, int(max(b['y_max'] for b in row))),
            }
            analysed = self.line_analyzer._analyze_single_line_crop(
                image, bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max'],
                0, 0, bbox['y_max'] - bbox['y_min']
            )
            line = analysed or {'bbox': bbox, 'raw_y_min': bbox['y_min'],
                                'raw_y_max': bbox['y_max'], 'ocr_text': '', 'ocr_conf': 0.0,
                                'latin_count': 0, 'devanagari_count': 0, 'total_chars': 0,
                                'latin_ratio': 0.0, 'deva_ratio': 0.0}
            line['bbox'] = bbox
            line['components'] = row
            line['height'] = bbox['y_max'] - bbox['y_min']
            line['width'] = bbox['x_max'] - bbox['x_min']
            line['raw_y_min'] = bbox['y_min']
            line['raw_y_max'] = bbox['y_max']
            line['flow_id'] = f'fallback-{len(lines)}'
            lines.append(line)
        return sorted(lines, key=lambda line: (line['bbox']['y_min'], line['bbox']['x_min']))

    def attach_page_features(self, image: np.ndarray, lines: List[Dict[str, Any]],
                             structure: Dict[str, Any], macro_proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach page-relative features and *same-flow* vertical gaps.

        The previous implementation measured gap to the previous line on the
        whole page. In multi-column manuscripts that can connect a left note to
        a center-body line at the same y coordinate. We instead use the closest
        horizontally compatible neighbour.
        """
        h, w = image.shape[:2]
        if not lines:
            return []

        heights = [max(1, line['height']) for line in lines]
        median_h = float(np.median(heights))
        ordered = sorted(lines, key=lambda line: ((line['bbox']['y_min'] + line['bbox']['y_max']) / 2.0,
                                                   line['bbox']['x_min']))

        for line in ordered:
            b = line['bbox']
            cx = (b['x_min'] + b['x_max']) / 2.0
            bw = max(1, b['x_max'] - b['x_min'])

            nearest_above = None
            nearest_below = None
            gap_above = float(h)
            gap_below = float(h)

            for other in ordered:
                if other is line:
                    continue
                ob = other['bbox']
                ocx = (ob['x_min'] + ob['x_max']) / 2.0
                obw = max(1, ob['x_max'] - ob['x_min'])
                union_w = max(1, max(b['x_max'], ob['x_max']) - min(b['x_min'], ob['x_min']))
                overlap = max(0, min(b['x_max'], ob['x_max']) - max(b['x_min'], ob['x_min'])) / union_w
                center_close = abs(cx - ocx) <= 0.30 * max(bw, obw)
                if overlap < 0.18 and not center_close:
                    continue

                if ob['y_max'] <= b['y_min']:
                    g = b['y_min'] - ob['y_max']
                    if g < gap_above:
                        gap_above, nearest_above = g, other
                elif ob['y_min'] >= b['y_max']:
                    g = ob['y_min'] - b['y_max']
                    if g < gap_below:
                        gap_below, nearest_below = g, other

            visual_filler_prior = False
            for proposal in macro_proposals:
                if proposal.get('class') != 'filler':
                    continue
                # Only explicitly visual macro sources are trusted here.
                if proposal.get('macro_source') not in {'image', 'seal', 'chart',
                                                         'formula_number', 'inline_formula'}:
                    continue
                if self._bbox_overlap_fraction(b, proposal.get('bbox', {})) >= 0.40:
                    visual_filler_prior = True
                    break

            line['features'] = {
                'x_center_norm': cx / max(1, w),
                'y_center_norm': ((b['y_min'] + b['y_max']) / 2.0) / max(1, h),
                'width_norm': (b['x_max'] - b['x_min']) / max(1, w),
                'height_norm': (b['y_max'] - b['y_min']) / max(1, h),
                'margin_distance_norm': min(b['x_min'], w - b['x_max']) / max(1, w),
                'gap_above_norm': gap_above / max(1, h),
                'gap_below_norm': gap_below / max(1, h),
                'median_line_height_norm': median_h / max(1, h),
                'filler_prior': visual_filler_prior,
            }
            line['visual_filler_prior'] = visual_filler_prior
            line['gap_above'] = int(gap_above)
            line['gap_below'] = int(gap_below)
            line['rel_pos'] = line['features']['y_center_norm']

        return ordered

    @staticmethod
    def _bbox_overlap_fraction(a: Dict[str, int], b: Dict[str, int]) -> float:
        if not a or not b:
            return 0.0
        ix1, iy1 = max(a['x_min'], b['x_min']), max(a['y_min'], b['y_min'])
        ix2, iy2 = min(a['x_max'], b['x_max']), min(a['y_max'], b['y_max'])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = max(1, (a['x_max'] - a['x_min']) * (a['y_max'] - a['y_min']))
        return inter / area

    def _has_positive_filler_evidence(self, line: Dict[str, Any]) -> bool:
        """High-precision filler gate.  Geometry never creates text filler.

        Filler is text only when OCR provides positive English/attribution
        evidence.  This deliberately ignores page position and raw Latin
        character counts because Indic OCR can emit ASCII-like garbage for
        Devanagari manuscripts.
        """
        positive_english, evidence = self._english_evidence(line)
        if positive_english:
            return True
        if line.get('visual_filler_prior') is True:
            return True

        text = (line.get('ocr_text') or '').strip()
        compact = ''.join(text.split())
        width_norm = float(line.get('features', {}).get('width_norm', 1.0) or 1.0)
        isolated_symbol = (
            bool(compact) and len(compact) <= 4 and
            not any(char.isalnum() for char in compact) and
            width_norm <= 0.12
        )
        return isolated_symbol

    def _flow_groups(self, lines: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group non-filler lines by normalized alignment and adaptive vertical cadence."""
        if not lines:
            return []
        median_h = float(np.median([max(1, line['height']) for line in lines]))
        gaps = [max(0, lines[i + 1]['bbox']['y_min'] - lines[i]['bbox']['y_max']) for i in range(len(lines) - 1)]
        useful_gaps = [gap for gap in gaps if gap > 0]
        median_gap = float(np.median(useful_gaps)) if useful_gaps else median_h * 0.6
        max_gap = max(2.5 * median_gap, 2.0 * median_h)
        flows: List[List[Dict[str, Any]]] = []
        for line in sorted(lines, key=lambda item: item['bbox']['y_min']):
            b = line['bbox']
            cx = (b['x_min'] + b['x_max']) / 2.0
            best_flow = None
            best_distance = float('inf')
            for flow in flows:
                previous = flow[-1]['bbox']
                prev_cx = (previous['x_min'] + previous['x_max']) / 2.0
                gap = b['y_min'] - previous['y_max']
                span = max(1, max(b['x_max'], previous['x_max']) - min(b['x_min'], previous['x_min']))
                overlap = max(0, min(b['x_max'], previous['x_max']) - max(b['x_min'], previous['x_min'])) / span
                if gap <= max_gap and (overlap >= 0.20 or abs(cx - prev_cx) <= 0.18 * span):
                    if abs(cx - prev_cx) < best_distance:
                        best_flow, best_distance = flow, abs(cx - prev_cx)
            if best_flow is None:
                flows.append([line])
            else:
                best_flow.append(line)
        return flows

    @staticmethod
    def _intersection_over_area(a: Dict[str, int], b: Dict[str, int]) -> float:
        if not a or not b:
            return 0.0
        ix1 = max(a['x_min'], b['x_min'])
        iy1 = max(a['y_min'], b['y_min'])
        ix2 = min(a['x_max'], b['x_max'])
        iy2 = min(a['y_max'], b['y_max'])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area = max(1, (a['x_max'] - a['x_min']) * (a['y_max'] - a['y_min']))
        return inter / area

    def _macro_role_overlap(self, bbox: Dict[str, int], macro_proposals: List[Dict[str, Any]],
                            classes: set) -> float:
        best = 0.0
        for proposal in macro_proposals or []:
            if proposal.get('class') not in classes and proposal.get('macro_source') not in classes:
                continue
            best = max(best, self._intersection_over_area(bbox, proposal.get('bbox', {})))
        return best

    def _fallback_macro_body_regions(self, image: np.ndarray, macro_proposals: List[Dict[str, Any]],
                                     current_regions: List[Dict[str, Any]],
                                     page_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        # Recover a missed dominant body from a coarse macro proposal when fine OCR
        # clearly under-detected the dense manuscript text.
        h, w = page_shape
        mains = [r for r in current_regions if r.get('class') == 'main_text']
        current_height = max((r['bbox']['y_max'] for r in mains), default=0) - min((r['bbox']['y_min'] for r in mains), default=h) if mains else 0
        current_ratio = current_height / max(1, h)
        if current_ratio >= self.macro_body_fallback_min_height_ratio:
            return []

        from src.postprocessor import tighten_bbox_to_content
        fallback = []
        for proposal in macro_proposals or []:
            if proposal.get('class') != 'main_text':
                continue
            b = proposal.get('bbox', {})
            x1, y1 = max(0, int(b.get('x_min', 0))), max(0, int(b.get('y_min', 0)))
            x2, y2 = min(w, int(b.get('x_max', 0))), min(h, int(b.get('y_max', 0)))
            if x2 <= x1 or y2 <= y1:
                continue
            bh = y2 - y1
            bw = x2 - x1
            if bh / max(1, h) < self.macro_body_fallback_min_height_ratio:
                continue
            if bw / max(1, w) < 0.45:
                continue
            cx = 0.5 * (x1 + x2) / max(1, w)
            if cx < 0.18 or cx > 0.82:
                continue
            tight, density = tighten_bbox_to_content(image, {'x_min': x1, 'y_min': y1, 'x_max': x2, 'y_max': y2}, pad=3)
            if density < self.macro_body_fallback_min_ink_density:
                continue
            fallback.append({
                'bbox': tight,
                'class': 'main_text',
                'confidence': round(float(np.clip(proposal.get('confidence', 0.60), 0.30, 0.90)), 4),
                'ocr_conf': float(proposal.get('confidence', 0.60)),
                'density': round(float(density), 4),
                'geometry_source': 'macro_body_fallback',
                'macro_source': proposal.get('macro_source'),
            })
        return fallback

    def assign_ownership(self, lines: List[Dict[str, Any]], structure: Dict[str, Any],
                         page_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Assign exactly one semantic owner to every OCR line.

        The decision is relational and page-adaptive:
        * main_text = persistent, width-stable reading flow(s)
        * side_text = narrow flow laterally adjacent to body and vertically
          overlapping it
        * header/footer = compact flow separated from the body above/below
        * filler = only positive semantic/visual evidence

        There is intentionally no rule such as ``y > 0.82 => footer`` or
        ``small/Latin-looking => filler``.
        """
        if not lines:
            return []

        h, w = page_shape
        metrics = structure.get('metrics', {}) if structure else {}
        h_line = float(metrics.get('h_line', np.median([max(1, l.get('height', 1)) for l in lines]) or 12.0))
        g_vert = float(metrics.get('g_vert', max(4.0, 0.75 * h_line)))
        frame = structure.get('content_frame', {
            'x_min': 0, 'y_min': 0, 'x_max': w, 'y_max': h,
        })
        macro_proposals = list(structure.get('macro_proposals', []) or [])

        # Ensure all lines have a usable flow identity.
        missing = [line for line in lines if line.get('flow_id') is None]
        if missing:
            existing = {line.get('flow_id') for line in lines if line.get('flow_id') is not None}
            for idx, flow in enumerate(self._flow_groups(missing)):
                flow_id = f'fallback-flow-{idx}'
                while flow_id in existing:
                    idx += 1
                    flow_id = f'fallback-flow-{idx}'
                for line in flow:
                    line['flow_id'] = flow_id
                existing.add(flow_id)

        # Filler is removed before body-flow discovery.  This prevents English
        # source/attribution lines from inflating the body frame or splitting it.
        for line in lines:
            line['filler_evidence'] = bool(self._has_positive_filler_evidence(line))
        filler_lines = [line for line in lines if line['filler_evidence']]
        manuscript_lines = [line for line in lines if not line['filler_evidence']]

        flows = self._owned_flow_groups(manuscript_lines)
        summaries = [self._flow_summary(flow, structure, page_shape) for flow in flows]
        if not summaries:
            for line in filler_lines:
                line['owner'] = 'filler'
                line['owner_evidence'] = {'flow_role': 'filler', 'filler': True}
                line['owner_score'] = 0.96
            return lines

        frame_h = max(1, frame['y_max'] - frame['y_min'])
        frame_w = max(1, frame['x_max'] - frame['x_min'])

        def script_ratio(summary):
            vals = [float(line.get('deva_ratio', 0.0) or 0.0) for line in summary['lines']]
            return float(np.mean(vals)) if vals else 0.0

        def flow_quality(summary):
            rows = int(summary['row_count'])
            width = float(summary['width_norm'])
            coverage = float((summary['bbox']['y_max'] - summary['bbox']['y_min']) / frame_h)
            persistence = min(1.0, rows / 5.0)
            width_stability = float(np.clip(
                np.median([
                    (line['bbox']['x_max'] - line['bbox']['x_min']) / frame_w
                    for line in summary['lines']
                ]) / max(width, 1e-6), 0.65, 1.0
            ))
            deva = script_ratio(summary)
            # Devanagari supports manuscript-body identity, but absence of it
            # does not disqualify OCR-damaged/Latin-script manuscripts.
            script_factor = 0.90 + 0.20 * deva
            top_singleton_penalty = 0.35 if (rows == 1 and
                summary['bbox']['y_min'] <= frame['y_min'] + 0.12 * frame_h) else 1.0
            bottom_singleton_penalty = 0.45 if (rows == 1 and
                summary['bbox']['y_max'] >= frame['y_max'] - 0.10 * frame_h) else 1.0
            return ((0.45 * persistence + 0.30 * np.clip(width, 0.0, 1.0) +
                     0.25 * np.clip(coverage, 0.0, 1.0)) * width_stability *
                    script_factor * top_singleton_penalty * bottom_singleton_penalty)

        for summary in summaries:
            summary['deva_ratio'] = script_ratio(summary)
            summary['quality'] = flow_quality(summary)

        # Prefer persistent flows. A one-line title or folio cannot outrank an
        # established body simply because it is wide.
        persistent = [s for s in summaries if s['row_count'] >= self.minimum_body_flow_lines]
        primary = max(persistent or summaries, key=lambda s: s['quality'])

        def box_vertical_overlap(a, b):
            return max(0, min(a['y_max'], b['y_max']) - max(a['y_min'], b['y_min'])) / max(1, min(a['y_max']-a['y_min'], b['y_max']-b['y_min']))

        body_ids = {primary['id']}
        primary_box = primary['bbox']
        primary_width = max(primary['width_norm'], 1e-6)
        for candidate in summaries:
            if candidate['id'] == primary['id']:
                continue
            b = candidate['bbox']
            rows_ok = candidate['row_count'] >= max(2, self.minimum_body_flow_lines)
            width_ok = candidate['width_norm'] >= self.body_width_min_ratio * primary_width
            score_ok = candidate['quality'] >= self.body_flow_score_ratio * max(primary['quality'], 1e-6)
            y_overlap = max(0, min(b['y_max'], primary_box['y_max']) - max(b['y_min'], primary_box['y_min'])) / max(1, min(b['y_max']-b['y_min'], primary_box['y_max']-primary_box['y_min']))
            # Multiple body columns are allowed when their vertical footprints
            # substantially overlap. A top/bottom isolated flow is not body.
            if rows_ok and width_ok and score_ok and y_overlap >= 0.45:
                body_ids.add(candidate['id'])

        # Merge nested OCR fragments that materially overlap selected body flows,
        # then rebuild summaries so role decisions use the coalesced geometry.
        body_ids = self._coalesce_overlapping_body_flows(lines, summaries, body_ids)
        flows = self._owned_flow_groups(manuscript_lines)
        summaries = [self._flow_summary(flow, structure, page_shape) for flow in flows]
        summary_by_id = {s['id']: s for s in summaries}
        body_summaries = [s for s in summaries if s['id'] in body_ids]
        if not body_summaries:
            body_summaries = [primary]
            body_ids = {primary['id']}

        body_x1 = min(s['bbox']['x_min'] for s in body_summaries)
        body_x2 = max(s['bbox']['x_max'] for s in body_summaries)
        body_y1 = min(s['bbox']['y_min'] for s in body_summaries)
        body_y2 = max(s['bbox']['y_max'] for s in body_summaries)
        body_h = max(1, body_y2 - body_y1)
        body_w = max(1, body_x2 - body_x1)
        body_center = 0.5 * (body_x1 + body_x2)
        median_body_line_width = float(np.median([
            line['bbox']['x_max'] - line['bbox']['x_min']
            for s in body_summaries for line in s['lines']
        ])) if body_summaries else body_w
        role_gap = max(1.35 * h_line, 1.35 * g_vert)

        flow_roles: Dict[Any, str] = {flow_id: 'main_text' for flow_id in body_ids}
        role_evidence: Dict[Any, Dict[str, Any]] = {}
        header_line_ids = set()
        footer_line_ids = set()
        macro_header_classes = {'header', 'doc_title', 'paragraph_title'}

        # Split an isolated top prefix from a body flow when a title was close
        # enough to the first body row to be linked into the same OCR flow.
        for summary in body_summaries:
            flow_lines = sorted(summary['lines'], key=lambda l: l['bbox']['y_min'])
            if len(flow_lines) < 2:
                continue
            prefix = []
            for idx, line in enumerate(flow_lines[:self.header_prefix_max_lines]):
                b = line['bbox']
                next_line = flow_lines[idx + 1] if idx + 1 < len(flow_lines) else None
                if next_line is None:
                    break
                gap = next_line['bbox']['y_min'] - b['y_max']
                width = b['x_max'] - b['x_min']
                median_body_width = max(1.0, float(np.median([l['bbox']['x_max'] - l['bbox']['x_min'] for l in flow_lines])))
                macro_overlap = self._macro_role_overlap(b, macro_proposals, macro_header_classes)
                title_like = (macro_overlap >= self.macro_header_overlap_ratio
                              or width <= 0.72 * median_body_width
                              or line['height'] >= 1.15 * h_line)
                separated = gap >= max(self.header_prefix_gap_ratio * g_vert, 0.90 * h_line)
                top_band = ((b['y_min'] - frame['y_min']) / max(1, frame['y_max'] - frame['y_min'])) <= 0.16
                non_filler = not line.get('filler_evidence', False)
                if idx == 0 and top_band and title_like and separated and non_filler:
                    prefix.append(line)
                elif prefix and title_like and separated and non_filler:
                    prefix.append(line)
                else:
                    break
            for line in prefix[:self.header_prefix_max_lines]:
                header_line_ids.add(id(line))

        for summary in summaries:
            flow_id = summary['id']
            bbox = summary['bbox']
            if flow_id in body_ids:
                role_evidence[flow_id] = {'body_flow': True, 'quality': summary.get('score', 0.0)}
                flow_roles[flow_id] = 'main_text'
                continue

            rows = int(summary['row_count'])
            width = max(1, bbox['x_max'] - bbox['x_min'])
            cx = 0.5 * (bbox['x_min'] + bbox['x_max'])
            header_gap = body_y1 - bbox['y_max']
            footer_gap = bbox['y_min'] - body_y2
            center_close = abs(cx - body_center) <= max(1.5 * h_line, 0.22 * body_w)
            vertical_overlap = max(0, min(bbox['y_max'], body_y2) - max(bbox['y_min'], body_y1)) / body_h
            lateral_left = bbox['x_max'] <= body_x1 + 1.5 * h_line
            lateral_right = bbox['x_min'] >= body_x2 - 1.5 * h_line
            narrow_side = width <= max(2.0 * h_line, self.side_width_ratio * median_body_line_width)
            side_like = (
                vertical_overlap >= 0.25 and rows >= 2 and narrow_side and (lateral_left or lateral_right)
            )
            if not side_like and vertical_overlap >= 0.35 and narrow_side and (lateral_left or lateral_right):
                side_like = True

            separator_above = self._separator_between(structure, bbox['y_max'], body_y1, 'horizontal')
            separator_below = self._separator_between(structure, body_y2, bbox['y_min'], 'horizontal')
            compact_header = rows <= self.header_max_lines
            compact_footer = rows <= self.footer_max_lines
            above_body = header_gap >= role_gap
            below_body = footer_gap >= role_gap

            macro_header_overlap = self._macro_role_overlap(bbox, macro_proposals, macro_header_classes)
            macro_footer_overlap = self._macro_role_overlap(bbox, macro_proposals, {'footer', 'number', 'footnote'})
            flow_height_ratio = (bbox['y_max'] - bbox['y_min']) / max(1, frame_h)
            header_height_ok = flow_height_ratio <= min(0.08, 2.8 * h_line / max(1, frame_h))
            footer_height_ok = flow_height_ratio <= self.footer_max_height_ratio
            header_like = compact_header and above_body and header_height_ok and (
                center_close or separator_above or width <= 0.95 * body_w or macro_header_overlap >= self.macro_header_overlap_ratio)
            footer_like = compact_footer and below_body and footer_height_ok and (
                center_close or separator_below or width <= 0.70 * body_w or macro_footer_overlap >= 0.35)

            body_overlap_height = max(0, min(bbox['y_max'], body_y2) - max(bbox['y_min'], body_y1))
            body_overlap_ratio = body_overlap_height / max(1, min(bbox['y_max'] - bbox['y_min'], body_h))
            if body_overlap_ratio > self.footer_max_body_overlap_ratio:
                footer_like = False

            # An oversized non-body OCR container that intersects the established
            # body is ambiguous rather than a legitimate semantic region. Drop it
            # instead of relabelling it as main_text/side_text. Macro-body fallback
            # can recover the true body when fine OCR under-detected it.
            oversized_overlap = (
                flow_height_ratio > max(self.footer_max_height_ratio, 0.08)
                and body_overlap_ratio > 0.25
                and width >= 0.45 * body_w
            )

            # A small isolated title/folio can sit closer than the median role
            # gap when an explicit rule separates it from the body.
            if separator_above and compact_header and bbox['y_max'] <= body_y1:
                header_like = True
            if separator_below and compact_footer and bbox['y_min'] >= body_y2:
                footer_like = True

            # Side annotations take precedence over header/footer only when they
            # truly overlap the body vertically and live laterally beside it.
            if side_like:
                role = 'side_text'
            elif header_like:
                role = 'header'
            elif footer_like:
                role = 'footer'
                for line in summary['lines']:
                    footer_line_ids.add(id(line))
            else:
                # A non-body flow inside the same vertical body span is retained
                # as main_text only if it is reasonably wide; otherwise it is a
                # side annotation that lacks a perfect edge alignment.
                if oversized_overlap:
                    role = 'ignore'
                else:
                    broad_enough = width >= self.body_width_min_ratio * median_body_line_width
                    role = 'main_text' if (vertical_overlap >= 0.45 and broad_enough) else 'side_text' if vertical_overlap >= 0.20 and narrow_side else 'main_text'

            flow_roles[flow_id] = role
            role_evidence[flow_id] = {
                'rows': rows, 'header_gap': float(header_gap), 'footer_gap': float(footer_gap),
                'vertical_overlap': float(vertical_overlap), 'center_close': bool(center_close),
                'side_like': bool(side_like), 'role_gap': float(role_gap),
            }

        # Every source text line gets exactly one owner. Filler evidence wins
        # because it is explicitly orthogonal to document geometry.
        for line in lines:
            flow_id = line.get('flow_id')
            if line.get('filler_evidence'):
                owner = 'filler'
                score = 0.96
                evidence = {'filler': True}
            elif id(line) in header_line_ids:
                owner = 'header'
                score = 0.88
                evidence = {'header_prefix': True, 'macro_header_overlap': self._macro_role_overlap(line['bbox'], macro_proposals, macro_header_classes)}
            elif id(line) in footer_line_ids:
                owner = 'footer'
                score = 0.84
                evidence = role_evidence.get(flow_id, {})
            else:
                owner = flow_roles.get(flow_id, 'main_text')
                score = 0.86 if owner == 'main_text' else 0.82
                evidence = role_evidence.get(flow_id, {})
            line['owner'] = owner
            line['owner_evidence'] = {**evidence, 'flow_role': owner}
            line['owner_score'] = score

        return lines

    @staticmethod
    def _owned_flow_groups(lines: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        for line in lines:
            groups.setdefault(line.get('flow_id', id(line)), []).append(line)
        return list(groups.values())

    @staticmethod
    def _coalesce_overlapping_body_flows(lines: List[Dict[str, Any]],
                                         summaries: List[Dict[str, Any]],
                                         body_flow_ids: set) -> set:
        """Give materially overlapping body fragments one flow identity.

        Fine-grained OCR can report a broad line and a nested fragment of the
        same manuscript row.  They are not independent reading flows, so they
        must not yield overlapping main-text regions.  Separate columns have
        no substantial horizontal overlap and remain distinct.
        """
        selected = [summary for summary in summaries if summary['id'] in body_flow_ids]
        parents = {summary['id']: summary['id'] for summary in selected}

        def find(flow_id):
            while parents[flow_id] != flow_id:
                parents[flow_id] = parents[parents[flow_id]]
                flow_id = parents[flow_id]
            return flow_id

        for index, left in enumerate(selected):
            left_box = left['bbox']
            left_width = max(1, left_box['x_max'] - left_box['x_min'])
            left_height = max(1, left_box['y_max'] - left_box['y_min'])
            for right in selected[index + 1:]:
                right_box = right['bbox']
                right_width = max(1, right_box['x_max'] - right_box['x_min'])
                right_height = max(1, right_box['y_max'] - right_box['y_min'])
                horizontal_overlap = max(0, min(left_box['x_max'], right_box['x_max']) -
                                         max(left_box['x_min'], right_box['x_min']))
                vertical_overlap = max(0, min(left_box['y_max'], right_box['y_max']) -
                                       max(left_box['y_min'], right_box['y_min']))
                if (horizontal_overlap / min(left_width, right_width) >= 0.70 and
                        vertical_overlap / min(left_height, right_height) >= 0.20):
                    parents[find(right['id'])] = find(left['id'])

        remapped = {flow_id: find(flow_id) for flow_id in parents}
        for line in lines:
            flow_id = line.get('flow_id')
            if flow_id in remapped:
                line['flow_id'] = remapped[flow_id]
        return set(remapped.values())

    @staticmethod
    def _separator_between(structure: Dict[str, Any], start: int, end: int, axis: str) -> bool:
        low, high = sorted((start, end))
        return any(low <= value <= high for value in structure.get('separators', {}).get(axis, []))

    def _flow_summary(self, flow: List[Dict[str, Any]], structure: Dict[str, Any],
                      page_shape: Tuple[int, int]) -> Dict[str, Any]:
        boxes = [line['bbox'] for line in flow]
        bbox = {
            'x_min': min(box['x_min'] for box in boxes), 'y_min': min(box['y_min'] for box in boxes),
            'x_max': max(box['x_max'] for box in boxes), 'y_max': max(box['y_max'] for box in boxes),
        }
        frame = structure.get('content_frame', {'x_min': 0, 'y_min': 0, 'x_max': page_shape[1], 'y_max': page_shape[0]})
        frame_width = max(1, frame['x_max'] - frame['x_min'])
        frame_height = max(1, frame['y_max'] - frame['y_min'])
        width_norm = float(np.median([(line['bbox']['x_max'] - line['bbox']['x_min']) / frame_width for line in flow]))
        coverage = (bbox['y_max'] - bbox['y_min']) / frame_height
        return {
            'id': flow[0].get('flow_id', id(flow[0])), 'lines': flow, 'bbox': bbox,
            'row_count': len(flow), 'width_norm': width_norm,
            'score': len(flow) * width_norm * max(0.10, coverage),
        }

    def _group_owned_lines(self, lines: List[Dict[str, Any]], owner: str) -> List[List[Dict[str, Any]]]:
        selected = [line for line in lines if line.get('owner') == owner]
        if owner in {'main_text', 'side_text', 'header', 'footer'}:
            return self._owned_flow_groups(selected)
        groups: List[List[Dict[str, Any]]] = []
        # Filler can span several nearby rows (for example, a wrapped source
        # note), but those rows must come from one connected reading flow.
        # Joining all filler by y alone would bridge independent decorative
        # motifs across the page and recreate a broad synthetic region.
        for flow in self._owned_flow_groups(selected):
            flow_groups: List[List[Dict[str, Any]]] = []
            for line in sorted(flow, key=lambda item: item['bbox']['y_min']):
                if not flow_groups or line['bbox']['y_min'] - flow_groups[-1][-1]['bbox']['y_max'] > max(4, 1.5 * line['height']):
                    flow_groups.append([line])
                else:
                    flow_groups[-1].append(line)
            groups.extend(flow_groups)
        return groups

    def _region_from_owned_lines(self, image: np.ndarray, owner: str,
                                 lines: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not lines:
            return None
        h, w = image.shape[:2]
        boxes = [line['bbox'] for line in lines]
        pad = max(2, int(round(np.median([line['height'] for line in lines]) * self.owned_region_padding_ratio)))
        bbox = {'x_min': max(0, min(b['x_min'] for b in boxes) - pad),
                'y_min': max(0, min(b['y_min'] for b in boxes) - pad),
                'x_max': min(w, max(b['x_max'] for b in boxes) + pad),
                'y_max': min(h, max(b['y_max'] for b in boxes) + pad)}
        from src.postprocessor import tighten_bbox_to_content
        tight, density = tighten_bbox_to_content(image, bbox, pad=pad, text_boxes=boxes)
        if owner == 'filler' and density < 0.012:
            return None
        evidence = [line.get('owner_score', 0.0) for line in lines]
        return {'bbox': tight, 'class': owner,
                'confidence': round(float(np.clip(np.mean(evidence), 0.10, 0.99)), 4),
                'ocr_conf': round(float(np.mean([line.get('ocr_conf', 0.0) for line in lines])), 4),
                'density': round(float(density), 4), 'geometry_source': 'line_owned',
                'owned_lines': boxes,
                'ocr_text': ' '.join(line.get('ocr_text', '').strip() for line in lines if line.get('ocr_text', '').strip())}

    def rebuild_owned_regions(self, image: np.ndarray, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        regions = []
        for owner in ('header', 'main_text', 'footer', 'side_text', 'filler'):
            for group in self._group_owned_lines(lines, owner):
                region = self._region_from_owned_lines(image, owner, group)
                if region is not None:
                    regions.append(region)
        # Padding is allowed to approach another owned flow, never to make two
        # semantically distinct line sets overlap.  Trim padding only; owned
        # line bounds themselves are preserved.
        mains = [region for region in regions if region['class'] == 'main_text']
        fillers = [region for region in regions if region['class'] == 'filler']
        for main in mains:
            main_raw_y1 = min((box['y_min'] for box in main.get('owned_lines', [main['bbox']])), default=main['bbox']['y_min'])
            main_raw_y2 = max((box['y_max'] for box in main.get('owned_lines', [main['bbox']])), default=main['bbox']['y_max'])
            for filler in fillers:
                h_overlap = max(0, min(main['bbox']['x_max'], filler['bbox']['x_max']) -
                                   max(main['bbox']['x_min'], filler['bbox']['x_min']))
                if h_overlap <= 0:
                    continue
                filler_raw_y1 = min((box['y_min'] for box in filler.get('owned_lines', [filler['bbox']])), default=filler['bbox']['y_min'])
                filler_raw_y2 = max((box['y_max'] for box in filler.get('owned_lines', [filler['bbox']])), default=filler['bbox']['y_max'])

                main_cy = (main['bbox']['y_min'] + main['bbox']['y_max']) / 2.0
                filler_cy = (filler['bbox']['y_min'] + filler['bbox']['y_max']) / 2.0

                if filler_cy > main_cy:
                    # Filler is below main_text (e.g. bottom attribution/source line)
                    if main_raw_y2 <= filler_raw_y1:
                        boundary = (main_raw_y2 + filler_raw_y1) // 2
                        main['bbox']['y_max'] = min(main['bbox']['y_max'], boundary)
                        filler['bbox']['y_min'] = max(filler['bbox']['y_min'], boundary + 1)
                    if main['bbox']['y_max'] >= filler['bbox']['y_min']:
                        main['bbox']['y_max'] = max(main['bbox']['y_min'] + 5, filler['bbox']['y_min'] - 1)
                elif filler_cy < main_cy:
                    # Filler is above main_text (e.g. top attribution/header line)
                    if filler_raw_y2 <= main_raw_y1:
                        boundary = (filler_raw_y2 + main_raw_y1) // 2
                        filler['bbox']['y_max'] = min(filler['bbox']['y_max'], boundary)
                        main['bbox']['y_min'] = max(main['bbox']['y_min'], boundary + 1)
                    if main['bbox']['y_min'] <= filler['bbox']['y_max']:
                        main['bbox']['y_min'] = min(main['bbox']['y_max'] - 5, filler['bbox']['y_max'] + 1)
        return regions

    def refine_page(self, image: np.ndarray, text_boxes: List[Dict[str, Any]],
                    structure: Dict[str, Any], macro_proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Primary refinement entry point: component inventory -> ownership -> regions."""
        lines = self.build_page_inventory(image, text_boxes, structure)
        lines = self.attach_page_features(image, lines, structure, macro_proposals)
        owned = self.assign_ownership(lines, structure, image.shape[:2])
        regions = self.rebuild_owned_regions(image, owned)
        fallback = self._fallback_macro_body_regions(image, macro_proposals, regions, image.shape[:2])
        if fallback:
            regions.extend(fallback)
        return regions

    def refine_detections(self, image: np.ndarray,
                         detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Full semantic refinement pipeline on candidate detections:
        - Inspects candidate main_text regions
        - Runs line-level extraction
        - Classifies lines into manuscript vs filler
        - Recomputes main_text geometry excluding filler lines
        - Verifies candidate header/footer regions against English attribution
        """
        from copy import deepcopy
        from src.postprocessor import tighten_bbox_to_content

        h, w = image.shape[:2]
        logger.debug(f"[LAYOUT] PP-DocLayoutV2 detections: {len(detections)}")

        main_text_cands = [d for d in detections if d.get('class') == 'main_text']
        other_cands = [d for d in detections if d.get('class') != 'main_text']
        logger.debug(f"[REFINER] Main text candidates: {len(main_text_cands)}")

        refined: List[Dict[str, Any]] = []

        # 1. Process candidate header/footer/title/side_text to ensure attribution and English notes are classified as filler
        is_fast = getattr(self.line_analyzer, 'fast_mode', True)
        for d in other_cands:
            cls = d.get('class')
            bbox = d.get('bbox')
            lines = []
            if cls in ['header', 'footer', 'title']:
                try:
                    lines = self.line_analyzer.extract_lines(image, bbox)
                except Exception as e:
                    logger.debug(f"Exception extracting lines for {cls} in {bbox}: {e}")
                    lines = []

                if lines and any(self.calculate_filler_score(line) >= self.score_threshold for line in lines):
                    ocr_info = self.line_analyzer.analyze_region_ocr(image, bbox, pre_extracted_lines=lines)
                    refined.append({
                        'bbox': bbox,
                        'class': 'filler',
                        'confidence': round(d.get('confidence', 0.8), 4),
                        'ocr_conf': ocr_info.get('ocr_conf', 0.0),
                        **ocr_info,
                    })
                    logger.debug(f"[FILLER] Reclassified {cls} to filler: {bbox}")
                    continue

                d_copy = deepcopy(d)
                if lines:
                    ocr_info = self.line_analyzer.analyze_region_ocr(image, bbox, pre_extracted_lines=lines)
                    d_copy.update(ocr_info)
                refined.append(d_copy)

            elif cls == 'side_text':
                tb_list = d.get('text_boxes', [])
                is_filler = False
                ocr_info = {}

                # Check actual text boxes if present (avoids horizontal projection failure on sparse margin strips)
                if tb_list:
                    step = max(1, len(tb_list) // 4)
                    sample_boxes = tb_list[::step][:4]
                    sample_crops = [
                        image[b['y_min']:b['y_max'], b['x_min']:b['x_max']]
                        for b in sample_boxes
                        if b['y_max'] > b['y_min'] and b['x_max'] > b['x_min']
                    ]
                    if sample_crops and self.line_analyzer.recognizer is not None:
                        ocr_outputs = self.line_analyzer._batch_ocr(sample_crops)
                        for text, o_conf in ocr_outputs:
                            from src.line_analyzer import count_script_characters
                            n_lat, n_dev, n_tot = count_script_characters(text)
                            lat_ratio = (n_lat / n_tot) if n_tot > 0 else 0.0
                            dev_ratio = (n_dev / n_tot) if n_tot > 0 else 0.0
                            line_dict = {
                                'ocr_text': text,
                                'ocr_conf': o_conf,
                                'latin_ratio': lat_ratio,
                                'latin_count': n_lat,
                                'deva_ratio': dev_ratio,
                                'rel_pos': 0.1,
                                'gap_above': 20,
                                'gap_below': 20,
                            }
                            if self.calculate_filler_score(line_dict) >= self.score_threshold:
                                is_filler = True
                                ocr_info = {'ocr_text': text, 'ocr_conf': o_conf}
                                break

                if not is_filler:
                    try:
                        lines = self.line_analyzer.extract_lines(image, bbox, sample_only=is_fast)
                    except Exception as e:
                        logger.debug(f"Exception extracting lines for {cls} in {bbox}: {e}")
                        lines = []

                    if lines and any(self.calculate_filler_score(line) >= self.score_threshold for line in lines):
                        is_filler = True
                        ocr_info = self.line_analyzer.analyze_region_ocr(image, bbox, pre_extracted_lines=lines)

                if is_filler:
                    if tb_list:
                        tb_sorted = sorted(tb_list, key=lambda b: b['y_min'])
                        clusters = []
                        cur_cl = [tb_sorted[0]]
                        for b in tb_sorted[1:]:
                            if b['y_min'] - cur_cl[-1]['y_max'] <= 45:
                                cur_cl.append(b)
                            else:
                                clusters.append(cur_cl)
                                cur_cl = [b]
                        clusters.append(cur_cl)

                        for cl in clusters:
                            cl_box = {
                                'x_min': min(b['x_min'] for b in cl),
                                'y_min': min(b['y_min'] for b in cl),
                                'x_max': max(b['x_max'] for b in cl),
                                'y_max': max(b['y_max'] for b in cl),
                            }
                            refined.append({
                                'bbox': cl_box,
                                'class': 'filler',
                                'confidence': round(d.get('confidence', 0.8), 4),
                                'ocr_conf': ocr_info.get('ocr_conf', 0.0),
                                'text_boxes': cl,
                                **ocr_info,
                            })
                            logger.debug(f"[FILLER] Emitted split filler item: {cl_box}")
                    else:
                        refined.append({
                            'bbox': bbox,
                            'class': 'filler',
                            'confidence': round(d.get('confidence', 0.8), 4),
                            'ocr_conf': ocr_info.get('ocr_conf', 0.0),
                            **ocr_info,
                        })
                        logger.debug(f"[FILLER] Reclassified {cls} to filler: {bbox}")
                    continue

                d_copy = deepcopy(d)
                d_copy.pop('text_boxes', None)
                if not is_fast and 'ocr_text' not in d_copy:
                    ocr_info = self.line_analyzer.analyze_region_ocr(image, bbox)
                    d_copy.update(ocr_info)
                refined.append(d_copy)

            elif cls == 'filler':
                d_copy = deepcopy(d)
                if 'ocr_text' not in d_copy or not d_copy['ocr_text']:
                    ocr_info = self.line_analyzer.analyze_region_ocr(image, bbox, sample_only=is_fast)
                    d_copy.update(ocr_info)
                refined.append(d_copy)

            else:
                refined.append(deepcopy(d))

        # 2. Process candidate main_text regions with line-level analysis
        for m_idx, cand in enumerate(main_text_cands):
            bbox = cand['bbox']
            lines = []
            try:
                lines = self.line_analyzer.extract_lines(image, bbox)
            except Exception as e:
                logger.debug(f"Exception extracting lines for candidate {m_idx + 1}: {e}")
                lines = []

            logger.debug(f"[LINE] Candidate {m_idx + 1}: Detected {len(lines)} lines")

            if not lines:
                # If no individual lines detected, analyze entire ROI as a single line/crop
                whole_line = None
                try:
                    whole_line = self.line_analyzer._analyze_single_line_crop(
                        image, bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max'], 0, 0, bbox['y_max'] - bbox['y_min']
                    )
                except Exception as e:
                    logger.debug(f"Exception during single line crop: {e}")

                tight_box, dens = tighten_bbox_to_content(image, bbox)

                if whole_line and self.calculate_filler_score(whole_line) >= self.score_threshold:
                    refined.append({
                        'bbox': whole_line['bbox'],
                        'class': 'filler',
                        'confidence': round(whole_line.get('ocr_conf', 0.85), 4),
                        'ocr_text': whole_line.get('ocr_text', ''),
                        'ocr_conf': whole_line.get('ocr_conf', 0.0),
                        'deva_ratio': whole_line.get('deva_ratio', 0.0),
                        'latin_ratio': whole_line.get('latin_ratio', 0.0),
                    })
                    logger.debug(f"  [FILLER] Reclassified entire candidate to filler: {whole_line['bbox']}")
                elif dens < 0.015 and (not whole_line or not whole_line.get('ocr_text', '').strip()) and not cand.get('text_boxes'):
                    # Blank predicted text box with no ink / no text evidence -> rejected
                    logger.debug(f"  [REJECTED] Blank/no-ink candidate text box rejected: {cand['bbox']} (dens={dens:.4f})")
                else:
                    cand_copy = deepcopy(cand)
                    cand_copy['bbox'] = tight_box
                    cand_copy['density'] = round(dens, 4)
                    if not is_fast and 'ocr_text' not in cand_copy:
                        ocr_info = self.line_analyzer.analyze_region_ocr(image, tight_box)
                        cand_copy.update(ocr_info)
                    refined.append(cand_copy)
                    logger.debug(f"  [MAIN_TEXT] Retained candidate main_text: {tight_box}")
                continue

            filler_lines = []
            ms_lines = []

            for line in lines:
                f_score = self.calculate_filler_score(line)
                txt_summary = line.get('ocr_text', '').strip()[:40]
                if f_score >= self.score_threshold:
                    filler_lines.append(line)
                    logger.debug(f"  [FILLER] Line y={line['raw_y_min']}..{line['raw_y_max']} score={f_score:.2f} text='{txt_summary}'")
                else:
                    ms_lines.append(line)
                    logger.debug(f"  [MAIN_TEXT] Line y={line['raw_y_min']}..{line['raw_y_max']} score={f_score:.2f} text='{txt_summary}'")

            logger.debug(f"[FILLER] Candidate lines: {len(filler_lines)}")
            logger.debug(f"[MAIN_TEXT] Manuscript lines retained: {len(ms_lines)}")

            # Group contiguous filler lines into coherent attribution blocks
            if filler_lines:
                filler_groups = []
                cur_fg = [filler_lines[0]]
                for f_line in filler_lines[1:]:
                    prev_fl = cur_fg[-1]
                    f_gap = f_line['raw_y_min'] - prev_fl['raw_y_max']
                    if f_gap <= 30:  # standard paragraph line spacing
                        cur_fg.append(f_line)
                    else:
                        filler_groups.append(cur_fg)
                        cur_fg = [f_line]
                filler_groups.append(cur_fg)

                for fg in filler_groups:
                    x1 = min(l['bbox']['x_min'] for l in fg)
                    y1 = min(l['bbox']['y_min'] for l in fg)
                    x2 = max(l['bbox']['x_max'] for l in fg)
                    y2 = max(l['bbox']['y_max'] for l in fg)
                    max_conf = max(l.get('ocr_conf', 0.85) for l in fg)
                    fg_text = " ".join([l['ocr_text'].strip() for l in fg if l.get('ocr_text', '').strip()])
                    total_latin = sum(l.get('latin_count', 0) for l in fg)
                    total_deva = sum(l.get('devanagari_count', 0) for l in fg)
                    total_chars = sum(l.get('total_chars', 0) for l in fg)
                    fg_latin_ratio = round(total_latin / total_chars, 4) if total_chars > 0 else 0.0
                    fg_deva_ratio = round(total_deva / total_chars, 4) if total_chars > 0 else 0.0

                    refined.append({
                        'bbox': {'x_min': x1, 'y_min': y1, 'x_max': x2, 'y_max': y2},
                        'class': 'filler',
                        'confidence': round(max_conf, 4),
                        'ocr_text': fg_text,
                        'ocr_conf': round(max_conf, 4),
                        'deva_ratio': fg_deva_ratio,
                        'latin_ratio': fg_latin_ratio,
                    })

            # Recompute main_text geometry ONLY from manuscript lines
            if ms_lines:
                # Group contiguous manuscript lines
                # If separated by large gaps or intervening filler lines, create separate blocks
                groups = []
                cur_group = [ms_lines[0]]
                for line in ms_lines[1:]:
                    prev_line = cur_group[-1]
                    gap = line['raw_y_min'] - prev_line['raw_y_max']
                    has_intervening_filler = any(
                        prev_line['raw_y_max'] <= fl['raw_y_min'] and fl['raw_y_max'] <= line['raw_y_min']
                        for fl in filler_lines
                    )
                    if gap <= 40 and not has_intervening_filler:
                        cur_group.append(line)
                    else:
                        groups.append(cur_group)
                        cur_group = [line]
                if cur_group:
                    groups.append(cur_group)

                for grp in groups:
                    grp_y1 = min(l['raw_y_min'] for l in grp)
                    grp_y2 = max(l['raw_y_max'] for l in grp)
                    grp_x1 = min(l['bbox']['x_min'] for l in grp)
                    grp_x2 = max(l['bbox']['x_max'] for l in grp)

                    # Recompute and tighten bounding box around manuscript content only
                    raw_grp_box = {'x_min': grp_x1, 'y_min': grp_y1, 'x_max': grp_x2, 'y_max': grp_y2}
                    tight_box, dens = tighten_bbox_to_content(image, raw_grp_box)

                    # Verify no overlap with filler lines
                    for fl in filler_lines:
                        fb = fl['bbox']
                        # If filler is below this group, clamp group bottom to be strictly above filler
                        if fb['y_min'] >= tight_box['y_min'] and tight_box['y_max'] > fb['y_min']:
                            tight_box['y_max'] = max(tight_box['y_min'] + 4, fb['y_min'] - 2)
                        # If filler is above this group, clamp group top to be strictly below filler
                        if fb['y_max'] <= tight_box['y_max'] and tight_box['y_min'] < fb['y_max']:
                            tight_box['y_min'] = min(tight_box['y_max'] - 4, fb['y_max'] + 2)

                    grp_text = " ".join([l['ocr_text'].strip() for l in grp if l.get('ocr_text', '').strip()])
                    grp_confs = [l['ocr_conf'] for l in grp if l.get('ocr_conf', 0.0) > 0.0]
                    mean_grp_conf = float(np.mean(grp_confs)) if grp_confs else cand.get('confidence', 0.85)
                    total_latin = sum(l.get('latin_count', 0) for l in grp)
                    total_deva = sum(l.get('devanagari_count', 0) for l in grp)
                    total_chars = sum(l.get('total_chars', 0) for l in grp)
                    grp_latin_ratio = round(total_latin / total_chars, 4) if total_chars > 0 else 0.0
                    grp_deva_ratio = round(total_deva / total_chars, 4) if total_chars > 0 else 0.0

                    bh = tight_box['y_max'] - tight_box['y_min']
                    bw = tight_box['x_max'] - tight_box['x_min']
                    box_area = bw * bh
                    img_area = max(1, w * h)
                    cy = (tight_box['y_min'] + tight_box['y_max']) / 2.0
                    cx = (tight_box['x_min'] + tight_box['x_max']) / 2.0
                    is_marginal_or_small = (cy / h < 0.18 or cy / h > 0.82 or cx / w < 0.15 or cx / w > 0.85 or (box_area / img_area < 0.35))

                    has_text = bool(grp_text.strip()) or bool(cand.get('text_boxes'))
                    if dens < 0.015 and not has_text and is_marginal_or_small:
                        logger.debug(f"  [REJECTED] False main_text rejected: no ink/text evidence in margin/background (dens={dens:.4f})")
                        continue

                    if tight_box['x_max'] > tight_box['x_min'] and tight_box['y_max'] > tight_box['y_min']:
                        refined.append({
                            'bbox': tight_box,
                            'class': 'main_text',
                            'confidence': cand['confidence'],
                            'density': dens,
                            'ocr_text': grp_text,
                            'ocr_conf': round(mean_grp_conf, 4),
                            'deva_ratio': grp_deva_ratio,
                            'latin_ratio': grp_latin_ratio,
                        })

        logger.debug(f"[POST] Final regions: {len(refined)}")
        return refined

    def _inspect_and_split_main_text(self, image: np.ndarray,
                                     main_det: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Convenience method to inspect and split a single candidate main_text detection."""
        refined = self.refine_detections(image, [main_det])
        m_list = [d for d in refined if d.get('class') == 'main_text']
        f_list = [d for d in refined if d.get('class') == 'filler']
        return m_list, f_list
