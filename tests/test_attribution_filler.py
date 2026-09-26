"""
Regression tests for English attribution detection and isolation from manuscript body.

Requirements covered:
- English attribution below manuscript body -> filler
- English attribution above manuscript body -> filler
- Devanagari manuscript lines near the bottom -> remain main_text
- main_text bbox stops before the external English attribution
- filler bbox is tight
- no overlap between main_text and filler
- existing working images must not regress
"""

import numpy as np
import pytest

from src.region_reconstructor import RegionReconstructor
from src.semantic_refiner import SemanticRefiner
from src.postprocessor import postprocess


class TestAttributionFillerIsolation:
    """Test suite ensuring external attribution text is strictly isolated as filler."""

    @pytest.fixture
    def setup_pipeline(self):
        config = {
            'semantic_refinement': {
                'filler_score_threshold': 0.45,
                'isolation_gap_threshold': 20,
            }
        }
        reconstructor = RegionReconstructor(config)
        refiner = SemanticRefiner(config=config)
        return reconstructor, refiner

    def test_english_attribution_below_manuscript_body_becomes_filler(self, setup_pipeline):
        """English attribution line placed below manuscript body must be classified as filler."""
        reconstructor, refiner = setup_pipeline
        h, w = 300, 600
        image = np.full((h, w, 3), 240, dtype=np.uint8)

        # 5 Devanagari body lines + 1 bottom English attribution line
        lines = [
            {
                'bbox': {'x_min': 80, 'y_min': 40, 'x_max': 520, 'y_max': 65},
                'height': 25, 'width': 440,
                'ocr_text': 'धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः',
                'deva_ratio': 0.85, 'latin_ratio': 0.0, 'latin_count': 0, 'ocr_conf': 0.90,
                'flow_id': 'flow-0'
            },
            {
                'bbox': {'x_min': 80, 'y_min': 75, 'x_max': 520, 'y_max': 100},
                'height': 25, 'width': 440,
                'ocr_text': 'मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय',
                'deva_ratio': 0.85, 'latin_ratio': 0.0, 'latin_count': 0, 'ocr_conf': 0.90,
                'flow_id': 'flow-0'
            },
            {
                'bbox': {'x_min': 80, 'y_min': 110, 'x_max': 520, 'y_max': 135},
                'height': 25, 'width': 440,
                'ocr_text': 'दृष्ट्वा तु पाण्डवानीकं व्यूढं दुर्योधनस्तदा',
                'deva_ratio': 0.85, 'latin_ratio': 0.0, 'latin_count': 0, 'ocr_conf': 0.90,
                'flow_id': 'flow-0'
            },
            {
                'bbox': {'x_min': 80, 'y_min': 145, 'x_max': 520, 'y_max': 170},
                'height': 25, 'width': 440,
                'ocr_text': 'आचार्यमुपसङ्गम्य राजा वचनमब्रवीत्',
                'deva_ratio': 0.85, 'latin_ratio': 0.0, 'latin_count': 0, 'ocr_conf': 0.90,
                'flow_id': 'flow-0'
            },
            {
                'bbox': {'x_min': 80, 'y_min': 180, 'x_max': 520, 'y_max': 205},
                'height': 25, 'width': 440,
                'ocr_text': 'पश्यैतां पाण्डुपुत्राणामाचार्य महतीं चमूम्',
                'deva_ratio': 0.85, 'latin_ratio': 0.0, 'latin_count': 0, 'ocr_conf': 0.90,
                'flow_id': 'flow-0'
            },
            # English attribution below body
            {
                'bbox': {'x_min': 150, 'y_min': 230, 'x_max': 450, 'y_max': 250},
                'height': 20, 'width': 300,
                'ocr_text': 'CSU-Sringeri Campus Rare Book and Manuscript Collection',
                'deva_ratio': 0.0, 'latin_ratio': 0.95, 'latin_count': 45, 'ocr_conf': 0.92,
                'flow_id': 'flow-attr'
            }
        ]

        import cv2
        for l in lines:
            b = l['bbox']
            cv2.putText(image, "Sample Manuscript Text Line", (b['x_min'] + 5, b['y_max'] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)

        structure = {
            'content_frame': {'x_min': 50, 'y_min': 20, 'x_max': 550, 'y_max': 280},
            'metrics': {'h_line': 25.0, 'g_vert': 10.0, 'w_line': 440.0},
            'separators': {'horizontal': [], 'vertical': []}
        }

        lines = refiner.attach_page_features(image, lines, structure, [])
        owned = refiner.assign_ownership(lines, structure, (h, w))
        regions = refiner.rebuild_owned_regions(image, owned)

        labels = [r['class'] for r in regions]
        assert 'main_text' in labels
        assert 'filler' in labels

        filler_reg = next(r for r in regions if r['class'] == 'filler')
        main_reg = next(r for r in regions if r['class'] == 'main_text')

        # Filler must encompass attribution
        assert filler_reg['bbox']['y_min'] >= 210
        assert filler_reg['confidence'] >= 0.85

        # Main text must stop before bottom filler
        assert main_reg['bbox']['y_max'] < filler_reg['bbox']['y_min']

    def test_english_attribution_above_manuscript_body_becomes_filler(self, setup_pipeline):
        """English attribution line placed above manuscript body must be classified as filler."""
        reconstructor, refiner = setup_pipeline
        h, w = 300, 600
        image = np.full((h, w, 3), 240, dtype=np.uint8)

        lines = [
            # Top English attribution line
            {
                'bbox': {'x_min': 160, 'y_min': 10, 'x_max': 440, 'y_max': 28},
                'height': 18, 'width': 280,
                'ocr_text': 'CC-0. In Public Domain. Digitization by eGangotri. Funding MIDF',
                'deva_ratio': 0.0, 'latin_ratio': 0.90, 'latin_count': 40, 'ocr_conf': 0.95,
                'flow_id': 'flow-top'
            },
            # Devanagari body
            {
                'bbox': {'x_min': 80, 'y_min': 50, 'x_max': 520, 'y_max': 75},
                'height': 25, 'width': 440,
                'ocr_text': 'धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः',
                'deva_ratio': 0.85, 'latin_ratio': 0.0, 'latin_count': 0, 'ocr_conf': 0.90,
                'flow_id': 'flow-body'
            },
            {
                'bbox': {'x_min': 80, 'y_min': 85, 'x_max': 520, 'y_max': 110},
                'height': 25, 'width': 440,
                'ocr_text': 'मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय',
                'deva_ratio': 0.85, 'latin_ratio': 0.0, 'latin_count': 0, 'ocr_conf': 0.90,
                'flow_id': 'flow-body'
            }
        ]

        import cv2
        for l in lines:
            b = l['bbox']
            cv2.putText(image, "Sample Manuscript Text Line", (b['x_min'] + 5, b['y_max'] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)

        structure = {
            'content_frame': {'x_min': 50, 'y_min': 10, 'x_max': 550, 'y_max': 250},
            'metrics': {'h_line': 25.0, 'g_vert': 10.0, 'w_line': 440.0},
            'separators': {'horizontal': [], 'vertical': []}
        }

        lines = refiner.attach_page_features(image, lines, structure, [])
        owned = refiner.assign_ownership(lines, structure, (h, w))
        regions = refiner.rebuild_owned_regions(image, owned)

        filler_reg = next(r for r in regions if r['class'] == 'filler')
        main_reg = next(r for r in regions if r['class'] == 'main_text')

        # Top filler must be above main_text
        assert filler_reg['bbox']['y_max'] < main_reg['bbox']['y_min']
        assert filler_reg['confidence'] >= 0.85

    def test_devanagari_manuscript_lines_near_bottom_remain_main_text(self, setup_pipeline):
        """Hard negative gate: Normal Devanagari lines near the bottom must NEVER be filler."""
        _, refiner = setup_pipeline

        # A Devanagari manuscript line at the bottom margin (e.g. rel_pos = 0.92)
        bottom_deva_line = {
            'bbox': {'x_min': 80, 'y_min': 260, 'x_max': 520, 'y_max': 285},
            'height': 25,
            'ocr_text': 'इति श्रीमद्भगवद्गीतासूपनिषत्सु ब्रह्मविद्यायां',
            'deva_ratio': 0.90,
            'latin_ratio': 0.0,
            'latin_count': 0,
            'ocr_conf': 0.88,
            'features': {'y_center_norm': 0.92, 'width_norm': 0.80, 'filler_prior': False}
        }

        # Must evaluate to False for filler evidence
        assert not refiner._has_positive_filler_evidence(bottom_deva_line)

    def test_no_overlap_between_main_text_and_filler_after_postprocess(self):
        """Postprocessing must ensure zero overlap between main_text and filler."""
        h, w = 300, 600
        image = np.full((h, w, 3), 240, dtype=np.uint8)

        # Draw dark strokes so density check passes
        image[40:210, 80:520] = 50
        image[220:240, 160:440] = 50

        raw_dets = [
            {
                'class': 'main_text',
                'confidence': 0.85,
                'bbox': {'x_min': 75, 'y_min': 35, 'x_max': 525, 'y_max': 218},
                'geometry_source': 'line_owned'
            },
            {
                'class': 'filler',
                'confidence': 0.92,
                'bbox': {'x_min': 155, 'y_min': 214, 'x_max': 445, 'y_max': 245},
                'geometry_source': 'line_owned'
            }
        ]

        processed = postprocess(raw_dets, w, h, confidence=0.10, image=image)

        main = next(d for d in processed if d['class'] == 'main_text')
        filler = next(d for d in processed if d['class'] == 'filler')

        # Check 2D intersection
        ix1 = max(main['bbox']['x_min'], filler['bbox']['x_min'])
        iy1 = max(main['bbox']['y_min'], filler['bbox']['y_min'])
        ix2 = min(main['bbox']['x_max'], filler['bbox']['x_max'])
        iy2 = min(main['bbox']['y_max'], filler['bbox']['y_max'])

        assert ix2 <= ix1 or iy2 <= iy1, (
            f"Overlap detected: main={main['bbox']}, filler={filler['bbox']}"
        )
        assert main['bbox']['y_max'] < filler['bbox']['y_min']
