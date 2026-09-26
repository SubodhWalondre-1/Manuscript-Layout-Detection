"""
Unit tests for FullPageTextDetector and RegionReconstructor (Two-Source Architecture).
"""

import numpy as np
import pytest

from src.region_reconstructor import RegionReconstructor
from src.text_detector import FullPageTextDetector


class TestTwoSourceArchitecture:
    """Test suite for full-page text detection and region reconstruction."""

    def test_text_detector_interface_and_graceful_handling(self):
        """Verify FullPageTextDetector initializes and handles empty/blank images."""
        detector = FullPageTextDetector(device="cpu")
        # Empty image should return empty list
        assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []

    def test_region_reconstructor_multi_column_reconstruction(self):
        """Verify RegionReconstructor separates header, columns, footer, and filler."""
        reconstructor = RegionReconstructor()
        h, w = 1000, 800
        dummy_img = np.ones((h, w, 3), dtype=np.uint8) * 240

        # Simulate text boxes for all regions:
        # Header (y: 50..120)
        # Left commentary (x: 20..150, y: 150..850)
        # Center main text (x: 180..620, y: 150..850)
        # Right commentary (x: 650..780, y: 150..850)
        # Footer (y: 890..930)
        # Filler (y: 950..980)
        text_boxes = [
            {'x_min': 200, 'y_min': 60, 'x_max': 600, 'y_max': 100, 'score': 0.95},
            {'x_min': 30, 'y_min': 200, 'x_max': 140, 'y_max': 400, 'score': 0.90},
            {'x_min': 220, 'y_min': 200, 'x_max': 580, 'y_max': 700, 'score': 0.92},
            {'x_min': 660, 'y_min': 200, 'x_max': 770, 'y_max': 400, 'score': 0.88},
            {'x_min': 350, 'y_min': 895, 'x_max': 450, 'y_max': 925, 'score': 0.85},
            {'x_min': 100, 'y_min': 955, 'x_max': 700, 'y_max': 975, 'score': 0.75},
        ]

        # Simulate a giant PP-DocLayoutV2 macro container box
        macro_proposals = [{
            'class': 'main_text',
            'confidence': 0.88,
            'bbox': {'x_min': 20, 'y_min': 50, 'x_max': 780, 'y_max': 940},
        }]

        regions = reconstructor.reconstruct(dummy_img, text_boxes, macro_proposals)
        classes = {r['class'] for r in regions}

        # Assert header, footer, filler are properly classified
        assert 'header' in classes
        assert 'footer' in classes
        assert 'filler' in classes

        # Assert giant main_text macro box was suppressed
        giant_boxes = [
            r for r in regions
            if r['class'] == 'main_text' and (r['bbox']['y_max'] - r['bbox']['y_min']) > 800
        ]
        assert len(giant_boxes) == 0

    def test_region_reconstructor_single_column_preservation(self):
        """Verify RegionReconstructor preserves single-column manuscript layouts."""
        reconstructor = RegionReconstructor()
        h, w = 300, 800
        dummy_img = np.ones((h, w, 3), dtype=np.uint8) * 240

        # Simulate single central block
        text_boxes = [
            {'x_min': 100, 'y_min': 50, 'x_max': 700, 'y_max': 90, 'score': 0.92},
            {'x_min': 100, 'y_min': 100, 'x_max': 700, 'y_max': 140, 'score': 0.95},
            {'x_min': 100, 'y_min': 150, 'x_max': 700, 'y_max': 190, 'score': 0.94},
        ]

        macro_proposals = [{
            'class': 'main_text',
            'confidence': 0.95,
            'bbox': {'x_min': 90, 'y_min': 45, 'x_max': 710, 'y_max': 200},
        }]

        regions = reconstructor.reconstruct(dummy_img, text_boxes, macro_proposals)
        assert len(regions) >= 1
        assert regions[0]['class'] == 'main_text'
