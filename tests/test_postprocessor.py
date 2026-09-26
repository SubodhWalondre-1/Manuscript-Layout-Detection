"""
Unit tests for ConfidenceScorer and Improved Post-Processing Pipeline.

Covers:
1. Multi-signal evidence-based confidence computation (zero fabrication / no arbitrary minimums).
2. Dynamic weight normalization when signals are missing.
3. Conflict resolution before bbox tightening (container suppression, cross-class deduplication).
4. Boundary clamping and impossible coordinate elimination.
5. Preservation of legitimate multiple boxes of the same class.
6. Prevention of incorrect cross-class merging.
"""

import numpy as np
import pytest
from src.confidence_scorer import ConfidenceScorer
from src.postprocessor import (
    postprocess,
    resolve_conflicts,
    clip_boxes,
    apply_nms,
    merge_overlapping,
)


class TestConfidenceScorer:
    def test_multi_signal_confidence_calculation(self):
        """Verifies multi-signal weighted combination without arbitrary minimums."""
        scorer = ConfidenceScorer()

        # High-evidence detection with all 5 signals
        det = {
            'class': 'main_text',
            'bbox': {'x_min': 200, 'y_min': 200, 'x_max': 800, 'y_max': 800},
            'layout_conf': 0.90,
            'ocr_det_conf': 0.92,
            'ocr_conf': 0.88,
            'density': 0.20,
            'deva_ratio': 0.95,
            'latin_ratio': 0.0,
        }

        conf = scorer.compute_confidence(det)
        assert 0.85 <= conf <= 0.96
        # Confirm breakdown signals were recorded
        assert 'confidence_signals' in det
        assert 'layout' in det['confidence_signals']
        assert 'ocr_det' in det['confidence_signals']
        assert 'ocr_rec' in det['confidence_signals']
        assert 'geometry' in det['confidence_signals']
        assert 'semantic' in det['confidence_signals']

    def test_dynamic_weight_normalization_when_signals_missing(self):
        """When some signals are absent (e.g. non-text seal or pure OCR), weights dynamically normalize."""
        scorer = ConfidenceScorer()

        # Pure OCR region without layout prior
        det_ocr_only = {
            'class': 'side_text',
            'bbox': {'x_min': 50, 'y_min': 200, 'x_max': 150, 'y_max': 800},
            'ocr_det_conf': 0.80,
            'ocr_conf': 0.75,
            'density': 0.18,
            'deva_ratio': 0.90,
        }

        conf = scorer.compute_confidence(det_ocr_only)
        # Should be a normalized weighted average of available signals
        assert 0.70 <= conf <= 0.90

    def test_no_arbitrary_minimum_clamping(self):
        """Verifies low-evidence detections receive genuine low scores (no max(0.88, score))."""
        scorer = ConfidenceScorer()

        weak_det = {
            'class': 'main_text',
            'bbox': {'x_min': 100, 'y_min': 100, 'x_max': 200, 'y_max': 150},
            'layout_conf': 0.25,
            'ocr_det_conf': 0.30,
            'ocr_conf': 0.20,
            'density': 0.01,
            'deva_ratio': 0.10,
        }

        conf = scorer.compute_confidence(weak_det)
        # Must reflect low evidence, NOT forced up to >= 0.80
        assert conf < 0.45


class TestPostprocessingPipeline:
    def test_conflict_resolution_before_bbox_tightening(self):
        """Giant parent container must be suppressed during conflict resolution before tightening."""
        parent_main = {
            'class': 'main_text',
            'confidence': 0.85,
            'bbox': {'x_min': 50, 'y_min': 50, 'x_max': 950, 'y_max': 950},
        }
        child_header = {
            'class': 'header',
            'confidence': 0.88,
            'bbox': {'x_min': 100, 'y_min': 60, 'x_max': 900, 'y_max': 150},
        }
        child_footer = {
            'class': 'footer',
            'confidence': 0.82,
            'bbox': {'x_min': 300, 'y_min': 850, 'x_max': 700, 'y_max': 920},
        }
        child_side = {
            'class': 'side_text',
            'confidence': 0.86,
            'bbox': {'x_min': 60, 'y_min': 200, 'x_max': 200, 'y_max': 800},
        }

        dets = [parent_main, child_header, child_footer, child_side]
        resolved = resolve_conflicts(dets, w=1000, h=1000)

        classes = [d['class'] for d in resolved]
        # Giant parent main_text must have been suppressed
        assert 'main_text' not in classes
        assert 'header' in classes
        assert 'footer' in classes
        assert 'side_text' in classes

    def test_preserves_legitimate_multiple_boxes_of_same_class(self):
        """Two legitimate side_text columns (left & right) or distinct main_text blocks must NOT be merged."""
        left_side = {
            'class': 'side_text',
            'confidence': 0.89,
            'bbox': {'x_min': 20, 'y_min': 200, 'x_max': 180, 'y_max': 800},
        }
        right_side = {
            'class': 'side_text',
            'confidence': 0.90,
            'bbox': {'x_min': 820, 'y_min': 200, 'x_max': 980, 'y_max': 800},
        }
        center_main = {
            'class': 'main_text',
            'confidence': 0.92,
            'bbox': {'x_min': 200, 'y_min': 200, 'x_max': 800, 'y_max': 800},
        }

        dets = [left_side, right_side, center_main]
        output = postprocess(dets, w=1000, h=1000, confidence=0.15)

        side_texts = [d for d in output if d['class'] == 'side_text']
        main_texts = [d for d in output if d['class'] == 'main_text']

        # Both left and right side_text must be retained
        assert len(side_texts) == 2
        assert len(main_texts) == 1

    def test_different_semantic_regions_never_merged(self):
        """Adjacent header and main_text boxes must never be merged into one class."""
        header_det = {
            'class': 'header',
            'confidence': 0.88,
            'bbox': {'x_min': 100, 'y_min': 50, 'x_max': 900, 'y_max': 190},
        }
        main_det = {
            'class': 'main_text',
            'confidence': 0.91,
            'bbox': {'x_min': 100, 'y_min': 200, 'x_max': 900, 'y_max': 800},
        }

        output = postprocess([header_det, main_det], w=1000, h=1000, confidence=0.15)
        classes = {d['class'] for d in output}
        assert 'header' in classes
        assert 'main_text' in classes
        assert len(output) == 2

    def test_impossible_and_out_of_bounds_coordinates_eliminated(self):
        """Boxes with negative coordinates, zero area, or out-of-bounds must be clipped or eliminated."""
        invalid_boxes = [
            # Degenerate zero-width box
            {'class': 'main_text', 'confidence': 0.90, 'bbox': {'x_min': 100, 'y_min': 100, 'x_max': 100, 'y_max': 200}},
            # Inverted impossible coordinates
            {'class': 'main_text', 'confidence': 0.90, 'bbox': {'x_min': 300, 'y_min': 200, 'x_max': 200, 'y_max': 100}},
            # Valid box with out-of-bounds padding (should be clipped)
            {'class': 'header', 'confidence': 0.85, 'bbox': {'x_min': -20, 'y_min': -10, 'x_max': 1020, 'y_max': 150}},
        ]

        output = postprocess(invalid_boxes, w=1000, h=1000, confidence=0.10)
        # Degenerate & inverted boxes must be dropped; out-of-bounds box must be clipped inside [0, 1000]
        assert len(output) == 1
        b = output[0]['bbox']
        assert 0 <= b['x_min'] < b['x_max'] <= 1000
        assert 0 <= b['y_min'] < b['y_max'] <= 1000

    def test_tight_bbox_localization_rules(self):
        """Verifies tight bounding box extraction for header, footer, side_text, and filler."""
        import cv2
        # Synthetic test image with compact text
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.putText(img, "COMPACT HEADER", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "FOLIO 12", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Loose candidate header box covering entire top band
        loose_header = {'class': 'header', 'confidence': 0.88, 'bbox': {'x_min': 20, 'y_min': 20, 'x_max': 480, 'y_max': 150}}
        # Loose candidate footer box covering entire bottom strip
        loose_footer = {'class': 'footer', 'confidence': 0.82, 'bbox': {'x_min': 20, 'y_min': 400, 'x_max': 480, 'y_max': 490}}

        out = postprocess([loose_header, loose_footer], w=500, h=500, confidence=0.10, image=img)
        hdr = next(d for d in out if d['class'] == 'header')['bbox']
        ftr = next(d for d in out if d['class'] == 'footer')['bbox']

        # Header must be tightened tightly to "COMPACT HEADER" text, not entire band
        assert hdr['x_min'] > 70
        assert hdr['x_max'] < 400
        assert hdr['y_min'] > 45
        assert hdr['y_max'] < 110

        # Footer must be tightened tightly to "FOLIO 12", not entire bottom strip
        assert ftr['x_min'] > 150
        assert ftr['x_max'] < 350
        assert ftr['y_min'] > 420
        assert ftr['y_max'] < 480

    def test_split_multi_item_filler_separated_by_gaps(self):
        """If multiple separated filler items exist (gap > 45px), creates separate tight boxes instead of one giant box."""
        import cv2
        img = np.zeros((600, 300, 3), dtype=np.uint8)
        # Note 1 at top: y ~ 50
        cv2.putText(img, "Asiatic Society Note", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Note 2 at bottom: y ~ 500 (gap = 450px)
        cv2.putText(img, "Bengali Region Note", (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        tbs = [
            {'x_min': 18, 'y_min': 35, 'x_max': 220, 'y_max': 55, 'score': 0.9},
            {'x_min': 18, 'y_min': 485, 'x_max': 210, 'y_max': 505, 'score': 0.9},
        ]

        # One giant candidate filler box covering whole margin
        giant_filler = {
            'class': 'filler',
            'confidence': 0.85,
            'bbox': {'x_min': 10, 'y_min': 10, 'x_max': 250, 'y_max': 550},
        }

        out = postprocess([giant_filler], w=300, h=600, confidence=0.10, image=img, text_boxes=tbs)
        fillers = [d for d in out if d['class'] == 'filler']

        # Must be split into 2 separate items!
        assert len(fillers) == 2
        f1 = min(fillers, key=lambda d: d['bbox']['y_min'])['bbox']
        f2 = max(fillers, key=lambda d: d['bbox']['y_min'])['bbox']

        # Neither box should span the empty 430px gap in the middle
        assert f1['y_max'] < 100
        assert f2['y_min'] > 450
        assert f2['y_min'] - f1['y_max'] > 300

    def test_filler_overlapping_inside_main_text_corrected_or_split(self):
        """Regression test: filler inside or overlapping main_text causes main_text region to be corrected or split."""
        main_cand = {
            'class': 'main_text',
            'confidence': 0.90,
            'bbox': {'x_min': 100, 'y_min': 100, 'x_max': 900, 'y_max': 900},
        }
        filler_cand = {
            'class': 'filler',
            'confidence': 0.92,
            'bbox': {'x_min': 150, 'y_min': 450, 'x_max': 850, 'y_max': 520},
        }

        output = postprocess([main_cand, filler_cand], w=1000, h=1000, confidence=0.10)
        mains = [d for d in output if d['class'] == 'main_text']
        fillers = [d for d in output if d['class'] == 'filler']

        assert len(fillers) == 1
        assert len(mains) == 2

        fb = fillers[0]['bbox']
        for m in mains:
            mb = m['bbox']
            no_overlap = (mb['y_max'] < fb['y_min']) or (mb['y_min'] > fb['y_max'])
            assert no_overlap, f"main_text {mb} must not overlap filler {fb}"

