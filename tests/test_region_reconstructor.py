"""
Unit tests for the Assignment-Specific Layout Reconstruction Stage (RegionReconstructor).

Covers:
1. Dynamic adaptive metrics computation on varying page dimensions.
2. Multi-column main_text and side_text layout reconstruction without model priors.
3. Symmetric side_text discovery on left and right margins.
4. Independent footer derivation without relying on oversized main_text scans.
5. Multi-signal filler detection (decorative motifs, figures, seals, Latin script).
6. Suppression of giant container boxes.
"""

import numpy as np
import pytest
from src.region_reconstructor import RegionReconstructor


class TestRegionReconstructor:
    @pytest.fixture
    def reconstructor(self):
        return RegionReconstructor()

    def test_adaptive_document_metrics(self, reconstructor):
        """Verifies dynamic H_line and G_vert calculation across different image scales."""
        # Synthetic text boxes with line height ~20 and gap ~10
        text_boxes = [
            {'x_min': 100, 'y_min': 100, 'x_max': 500, 'y_max': 120, 'score': 0.95},
            {'x_min': 100, 'y_min': 130, 'x_max': 500, 'y_max': 150, 'score': 0.95},
            {'x_min': 100, 'y_min': 160, 'x_max': 500, 'y_max': 180, 'score': 0.95},
            {'x_min': 100, 'y_min': 190, 'x_max': 500, 'y_max': 210, 'score': 0.95},
        ]
        metrics = reconstructor.compute_adaptive_metrics(text_boxes, (1000, 800))
        assert 18.0 <= metrics['h_line'] <= 22.0
        assert 8.0 <= metrics['g_vert'] <= 12.0
        assert metrics['aspect_ratio'] == pytest.approx(800 / 1000)

        # Empty boxes should fallback to adaptive defaults scaled by page height
        empty_metrics = reconstructor.compute_adaptive_metrics([], (2000, 1000))
        assert empty_metrics['h_line'] >= 8.0
        assert empty_metrics['g_vert'] >= 4.0

    def test_structure_analysis_exposes_normalized_component_evidence(self, reconstructor):
        """The proposal layer supplies normalized structure rather than final labels."""
        image = np.full((600, 1200, 3), 240, dtype=np.uint8)
        boxes = [
            {'x_min': 100, 'y_min': 100, 'x_max': 500, 'y_max': 125, 'score': 0.9},
            {'x_min': 100, 'y_min': 145, 'x_max': 500, 'y_max': 170, 'score': 0.9},
            {'x_min': 700, 'y_min': 100, 'x_max': 1100, 'y_max': 125, 'score': 0.9},
            {'x_min': 700, 'y_min': 145, 'x_max': 1100, 'y_max': 170, 'score': 0.9},
        ]
        structure = reconstructor.analyze_structure(image, boxes, [])
        assert structure['metrics']['line_height_norm'] == pytest.approx(25 / 600)
        assert structure['metrics']['line_gap_norm'] > 0
        assert len(structure['components']) == len(boxes)
        assert all(0.0 <= item['x_center_norm'] <= 1.0 for item in structure['components'])

    def test_multi_column_main_text_support(self, reconstructor):
        """Verifies that multiple text columns or separate vertical blocks are preserved."""
        # Synthetic page (H=1000, W=1000) with two distinct vertical text blocks separated by a large gap
        img = np.full((1000, 1000, 3), 240, dtype=np.uint8)

        # Block 1 (upper body)
        upper_boxes = [
            {'x_min': 200, 'y_min': 200, 'x_max': 800, 'y_max': 230, 'score': 0.9},
            {'x_min': 200, 'y_min': 240, 'x_max': 800, 'y_max': 270, 'score': 0.9},
        ]
        # Block 2 (lower body, gap = 230px >> 3.5 * g_vert)
        lower_boxes = [
            {'x_min': 200, 'y_min': 500, 'x_max': 800, 'y_max': 530, 'score': 0.9},
            {'x_min': 200, 'y_min': 540, 'x_max': 800, 'y_max': 570, 'score': 0.9},
        ]
        all_boxes = upper_boxes + lower_boxes

        reconstructed = reconstructor.reconstruct(img, all_boxes, macro_proposals=[])
        main_regions = [r for r in reconstructed if r['class'] == 'main_text']
        # Should produce distinct blocks rather than a single collapsed giant box
        assert len(main_regions) == 2
        assert main_regions[0]['bbox']['y_max'] < main_regions[1]['bbox']['y_min']

    def test_symmetric_side_text_without_model_prior(self, reconstructor):
        """Verifies symmetric left and right side_text discovery via column gutters."""
        img = np.full((1000, 1200, 3), 240, dtype=np.uint8)

        # Construct 3 columns via text box distribution (occupancy guttering)
        # Left column (side_text): x in [50, 250]
        left_boxes = [
            {'x_min': 50, 'y_min': 300, 'x_max': 250, 'y_max': 330, 'score': 0.92},
            {'x_min': 50, 'y_min': 350, 'x_max': 250, 'y_max': 380, 'score': 0.92},
            {'x_min': 50, 'y_min': 400, 'x_max': 250, 'y_max': 430, 'score': 0.92},
        ]
        # Center column (main_text): x in [350, 850]
        center_boxes = [
            {'x_min': 350, 'y_min': 300, 'x_max': 850, 'y_max': 330, 'score': 0.95},
            {'x_min': 350, 'y_min': 350, 'x_max': 850, 'y_max': 380, 'score': 0.95},
            {'x_min': 350, 'y_min': 400, 'x_max': 850, 'y_max': 430, 'score': 0.95},
        ]
        # Right column (side_text): x in [950, 1150]
        right_boxes = [
            {'x_min': 950, 'y_min': 300, 'x_max': 1150, 'y_max': 330, 'score': 0.91},
            {'x_min': 950, 'y_min': 350, 'x_max': 1150, 'y_max': 380, 'score': 0.91},
            {'x_min': 950, 'y_min': 400, 'x_max': 1150, 'y_max': 430, 'score': 0.91},
        ]
        mid_boxes = left_boxes + center_boxes + right_boxes

        # No macro priors provided: pure geometric discovery
        reconstructed = reconstructor.reconstruct(img, mid_boxes, macro_proposals=[])

        side_texts = [r for r in reconstructed if r['class'] == 'side_text']
        main_texts = [r for r in reconstructed if r['class'] == 'main_text']

        assert len(side_texts) == 2, f"Expected 2 side_text regions (left & right), got {len(side_texts)}"
        assert len(main_texts) == 1, f"Expected 1 central main_text region, got {len(main_texts)}"

        # Verify left is to the left of main, and right is to the right of main
        left_side = min(side_texts, key=lambda r: r['bbox']['x_min'])
        right_side = max(side_texts, key=lambda r: r['bbox']['x_min'])
        main_box = main_texts[0]['bbox']

        assert left_side['bbox']['x_max'] <= main_box['x_min'] + 50
        assert right_side['bbox']['x_min'] >= main_box['x_max'] - 50

    def test_independent_footer_without_oversized_scan(self, reconstructor):
        """Verifies footer detection is derived independently and suppresses oversized containers."""
        img = np.full((1000, 1000, 3), 240, dtype=np.uint8)

        # Text boxes in middle body (y: 200..700) and bottom footer zone (y: 890..920)
        body_boxes = [
            {'x_min': 200, 'y_min': 250, 'x_max': 800, 'y_max': 280, 'score': 0.9},
            {'x_min': 200, 'y_min': 300, 'x_max': 800, 'y_max': 330, 'score': 0.9},
        ]
        footer_boxes = [
            {'x_min': 400, 'y_min': 900, 'x_max': 600, 'y_max': 920, 'score': 0.92},
        ]
        all_boxes = body_boxes + footer_boxes

        # An oversized macro proposal from PP-DocLayoutV2 that covers 90% of the page
        oversized_macro = [
            {
                'class': 'main_text',
                'confidence': 0.88,
                'bbox': {'x_min': 50, 'y_min': 50, 'x_max': 950, 'y_max': 950},
            }
        ]

        reconstructed = reconstructor.reconstruct(img, all_boxes, macro_proposals=oversized_macro)

        classes = [r['class'] for r in reconstructed]
        assert 'footer' in classes
        assert 'main_text' in classes

        # Giant container box covering 90% of the page should have been suppressed
        for r in reconstructed:
            area = (r['bbox']['x_max'] - r['bbox']['x_min']) * (r['bbox']['y_max'] - r['bbox']['y_min'])
            assert area / (1000 * 1000) < 0.80

    def test_multi_signal_filler_decorative_and_pencil(self, reconstructor):
        """Verifies multi-signal filler handling (macro decorative priors + margin text boxes)."""
        img = np.full((1000, 1000, 3), 240, dtype=np.uint8)

        # Text boxes in body + bottom margin (pencil / attribution)
        boxes = [
            {'x_min': 200, 'y_min': 250, 'x_max': 800, 'y_max': 280, 'score': 0.9},
            {'x_min': 100, 'y_min': 950, 'x_max': 700, 'y_max': 970, 'score': 0.85},
        ]
        # Decorative macro proposals (seal, figure, decoration)
        macro_props = [
            {
                'class': 'seal',
                'confidence': 0.91,
                'bbox': {'x_min': 850, 'y_min': 850, 'x_max': 950, 'y_max': 950},
            },
            {
                'class': 'figure',
                'confidence': 0.87,
                'bbox': {'x_min': 50, 'y_min': 100, 'x_max': 150, 'y_max': 200},
            },
        ]

        reconstructed = reconstructor.reconstruct(img, boxes, macro_proposals=macro_props)

        fillers = [r for r in reconstructed if r['class'] == 'filler']
        # Both decorative macro items and the bottom margin box become filler
        assert len(fillers) >= 2

    def test_two_separate_footer_regions_both_footer(self, reconstructor):
        """Regression test: multiple independent footer regions on the same page are all retained as footer."""
        img = np.full((1000, 1000, 3), 240, dtype=np.uint8)
        # Main text in central body
        body_boxes = [
            {'x_min': 200, 'y_min': 200, 'x_max': 800, 'y_max': 230, 'score': 0.95},
            {'x_min': 200, 'y_min': 250, 'x_max': 800, 'y_max': 280, 'score': 0.95},
        ]
        # Two separate footer regions: Left folio number (x: 100..200) and Right colophon marker (x: 750..850)
        footer_left = [
            {'x_min': 100, 'y_min': 900, 'x_max': 200, 'y_max': 930, 'score': 0.92},
        ]
        footer_right = [
            {'x_min': 750, 'y_min': 900, 'x_max': 850, 'y_max': 930, 'score': 0.91},
        ]
        all_boxes = body_boxes + footer_left + footer_right

        reconstructed = reconstructor.reconstruct(img, all_boxes, macro_proposals=[])
        footers = [r for r in reconstructed if r['class'] == 'footer']
        fillers = [r for r in reconstructed if r['class'] == 'filler']

        # Both independent footer regions must be preserved as footer
        assert len(footers) == 2, f"Expected 2 footers, got {len(footers)}"
        # Neither footer region should be converted to filler
        assert len(fillers) == 0

        # Ensure all footer boxes stay inside page boundaries [0, 1000]
        for f in footers:
            b = f['bbox']
            assert 0 <= b['x_min'] < b['x_max'] <= 1000
            assert 0 <= b['y_min'] < b['y_max'] <= 1000

    def test_structure_splits_same_baseline_into_interior_reading_flows(self, reconstructor):
        """Three columns remain independent even though every row shares y alignment."""
        image = np.full((900, 1200, 3), 235, dtype=np.uint8)
        boxes = []
        for y in (180, 230, 280, 330, 380):
            boxes.extend([
                {'x_min': 130, 'y_min': y, 'x_max': 290, 'y_max': y + 24, 'score': 0.9},
                {'x_min': 370, 'y_min': y, 'x_max': 830, 'y_max': y + 24, 'score': 0.95},
                {'x_min': 910, 'y_min': y, 'x_max': 1070, 'y_max': y + 24, 'score': 0.9},
            ])

        structure = reconstructor.analyze_structure(image, boxes, [])
        flows = sorted(structure['flows'], key=lambda flow: flow['bbox']['x_min'])

        assert len(flows) == 3
        assert flows[0]['row_count'] == flows[1]['row_count'] == flows[2]['row_count'] == 5
        assert flows[1]['median_width_norm'] > flows[0]['median_width_norm']
        assert flows[1]['median_width_norm'] > flows[2]['median_width_norm']
        assert structure['content_frame']['x_min'] < flows[0]['bbox']['x_min']
        assert structure['content_frame']['x_max'] > flows[2]['bbox']['x_max']
