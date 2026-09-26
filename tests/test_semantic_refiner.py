"""
Unit tests for Semantic Refiner and Line Analyzer.

Tests:
- English / Indic script scoring
- Multi-signal filler classification
- Manuscript body vs filler separation
- Main text recomputation and geometry exclusion
- Interrupted main_text splitting
- Ambiguous line handling
- Absent class guarantees
- Bounding box boundary clipping
"""

import os
import sys
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.line_analyzer import LineAnalyzer, count_script_characters
from src.semantic_refiner import SemanticRefiner
from src.postprocessor import clip_boxes
from src.output_handler import save_json_metadata
import tempfile
import json


class TestSemanticRefiner(unittest.TestCase):

    def setUp(self):
        self.refiner = SemanticRefiner()
        self.analyzer = LineAnalyzer()

    # 1. Script scoring tests
    def test_script_character_counting_english(self):
        """English attribution text should have high Latin ratio (>85%) and 0% Devanagari."""
        text = "Digitization by eGangotri. Funding MIDF. CC-0 In Public Domain."
        latin, deva, total = count_script_characters(text)
        self.assertGreater(latin, 30)
        self.assertEqual(deva, 0)
        self.assertGreater(latin / total, 0.85)

    def test_script_character_counting_devanagari(self):
        """Devanagari manuscript text should have 0% Latin and >80% Devanagari."""
        text = "॥ श्रीगणेशाय नमः ॥ अथ प्रथमोऽध्यायः । धर्मक्षेत्रे कुरुक्षेत्रे"
        latin, deva, total = count_script_characters(text)
        self.assertEqual(latin, 0)
        self.assertGreater(deva, 30)
        self.assertGreater(deva / total, 0.70)

    def test_script_character_counting_vedic_extensions(self):
        """Vedic extensions (\u1CD0-\u1CFF and \uA8E0-\uA8FF) must count as Indic/Devanagari."""
        text = "अ॒ग्निमी॑ळे पु॒रोहि॑तं"
        latin, deva, total = count_script_characters(text)
        self.assertEqual(latin, 0)
        self.assertGreater(deva, 5)

    # 2. Filler classification tests
    def test_filler_scoring_attribution_keywords(self):
        """Lines containing source/digitization keywords must yield high filler score."""
        line = {
            'ocr_text': 'CSU-Sringeri Campus Rare Book and Manuscript Collection, Sringeri',
            'ocr_conf': 0.98,
            'latin_count': 55,
            'latin_ratio': 1.0,
            'deva_ratio': 0.0,
            'gap_above': 25,
            'gap_below': 5,
            'rel_pos': 0.95,
            'height': 16,
        }
        score = self.refiner.calculate_filler_score(line)
        self.assertGreaterEqual(score, self.refiner.score_threshold)
        self.assertGreater(score, 0.70)

    def test_filler_scoring_public_domain_metadata(self):
        """Public domain and CC-0 metadata lines must be classified as filler."""
        line = {
            'ocr_text': 'CC-0. In Public Domain. Digitization by eGangotri.',
            'ocr_conf': 0.95,
            'latin_count': 35,
            'latin_ratio': 1.0,
            'deva_ratio': 0.0,
            'gap_above': 2,
            'gap_below': 20,
            'rel_pos': 0.05,
            'height': 14,
        }
        score = self.refiner.calculate_filler_score(line)
        self.assertGreaterEqual(score, self.refiner.score_threshold)

    def test_manuscript_body_scoring(self):
        """Genuine Devanagari manuscript body line must NOT be classified as filler."""
        line = {
            'ocr_text': 'धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः',
            'ocr_conf': 0.92,
            'latin_count': 0,
            'latin_ratio': 0.0,
            'deva_ratio': 0.90,
            'gap_above': 5,
            'gap_below': 5,
            'rel_pos': 0.50,
            'height': 22,
        }
        score = self.refiner.calculate_filler_score(line)
        self.assertLess(score, self.refiner.score_threshold)
        self.assertEqual(score, 0.0)

    def test_ambiguous_line_handling(self):
        """Ambiguous line with faint markings and no keywords is not forced to filler."""
        line = {
            'ocr_text': '123',
            'ocr_conf': 0.30,
            'latin_count': 0,
            'latin_ratio': 0.0,
            'deva_ratio': 0.0,
            'gap_above': 5,
            'gap_below': 5,
            'rel_pos': 0.50,
            'height': 15,
        }
        score = self.refiner.calculate_filler_score(line)
        self.assertLess(score, self.refiner.score_threshold)

    def test_devanagari_misrecognized_as_latin_not_filler(self):
        """OCR misreading Devanagari as Latin gibberish must NOT be filler,
        regardless of OCR confidence. Without attribution keywords, the score
        is hard-capped at 0.35 (below the 0.45 threshold).
        """
        # Case 1: Low confidence gibberish
        line_low = {
            'ocr_text': 'HaRRRaFTA a:ll 3TAT T:::1',
            'ocr_conf': 0.35,
            'latin_count': 15, 'latin_ratio': 0.75,
            'deva_ratio': 0.0,
            'gap_above': 15, 'gap_below': 15,
            'rel_pos': 0.40, 'height': 25,
        }
        score = self.refiner.calculate_filler_score(line_low)
        self.assertLessEqual(score, 0.35,
                             f"Low-conf gibberish must be capped (score={score:.2f})")

        # Case 2: HIGH confidence gibberish (still no keywords → still blocked)
        line_high = {
            'ocr_text': 'aaf ARraat aTri aat GAaRa',
            'ocr_conf': 0.75,
            'latin_count': 20, 'latin_ratio': 0.90,
            'deva_ratio': 0.0,
            'gap_above': 20, 'gap_below': 20,
            'rel_pos': 0.30, 'height': 20,
        }
        score = self.refiner.calculate_filler_score(line_high)
        self.assertLessEqual(score, 0.35,
                             f"High-conf gibberish without keywords must be capped (score={score:.2f})")

    def test_filler_scoring_strong_latin_without_keywords(self):
        """Marginal English text with strong Latin density but no keywords must be scored as filler."""
        line = {
            'ocr_text': 'Transcribed by J. Smith, Oxford University Press',
            'ocr_conf': 0.92,
            'latin_count': 38,
            'latin_ratio': 0.95,
            'deva_ratio': 0.0,
            'gap_above': 25,
            'gap_below': 5,
            'rel_pos': 0.94,
            'height': 16,
        }
        score = self.refiner.calculate_filler_score(line)
        self.assertGreaterEqual(score, self.refiner.score_threshold)

    # 3. Main text recomputation and filler exclusion
    def test_main_text_recomputation_excludes_bottom_filler(self):
        """Candidate main_text covering bottom attribution must be tightened to manuscript only."""
        img = np.full((400, 800, 3), 40, dtype=np.uint8)
        # Manuscript body at rows 60..250 with realistic line pitch
        img[60:250, 100:700] = 220
        cv2.putText(img, "Devanagari Manuscript Text Line One", (120, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
        cv2.putText(img, "Devanagari Manuscript Text Line Two", (120, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
        # English attribution line at rows 365..385
        cv2.putText(img, "CSU-Sringeri Campus Rare Book and Manuscript Collection",
                    (150, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)

        candidate = {
            'bbox': {'x_min': 100, 'y_min': 50, 'x_max': 700, 'y_max': 390},
            'class': 'main_text',
            'confidence': 0.85
        }

        # Mock recognizer for testing
        class MockRecognizer:
            def predict(self, crop):
                if np.mean(crop) < 100:
                    return [{'rec_text': 'CSU-Sringeri Campus Rare Book and Manuscript Collection', 'rec_score': 0.98}]
                return [{'rec_text': 'श्रीगणेशाय नमः', 'rec_score': 0.90}]

        refiner = SemanticRefiner(recognizer=MockRecognizer())
        m_list, f_list = refiner._inspect_and_split_main_text(img, candidate)

        self.assertEqual(len(m_list), 1)
        self.assertEqual(len(f_list), 1)
        self.assertEqual(m_list[0]['class'], 'main_text')
        self.assertEqual(f_list[0]['class'], 'filler')

        # Critical check: recomputed main_text MUST NOT contain bottom filler
        self.assertLess(m_list[0]['bbox']['y_max'], f_list[0]['bbox']['y_min'])
        self.assertLess(m_list[0]['bbox']['y_max'], 350)
        self.assertGreaterEqual(f_list[0]['bbox']['y_min'], 350)

    def test_main_text_interrupted_by_filler_splits_blocks(self):
        """When an attribution line interrupts the manuscript, two separate main_text regions are formed."""
        img = np.full((500, 800, 3), 40, dtype=np.uint8)
        # Top manuscript block
        img[50:180, 100:700] = 220
        cv2.putText(img, "Manuscript Section 1", (120, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
        # Intervening English line
        cv2.putText(img, "Digitized by eGangotri Archive",
                    (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)
        # Bottom manuscript block
        img[300:430, 100:700] = 220
        cv2.putText(img, "Manuscript Section 2", (120, 370),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

        candidate = {
            'bbox': {'x_min': 100, 'y_min': 40, 'x_max': 700, 'y_max': 440},
            'class': 'main_text',
            'confidence': 0.88
        }

        class MockRecognizer:
            def predict(self, crop):
                if 220 <= np.mean(crop) or (crop.shape[0] < 30 and 'Digitized' in ''):
                    pass
                # Detect the intervening line
                return [{'rec_text': 'Digitized by eGangotri Archive', 'rec_score': 0.95}]

        # Inject known lines directly to test splitting geometry
        filler_line = {
            'bbox': {'x_min': 140, 'y_min': 230, 'x_max': 500, 'y_max': 250},
            'raw_y_min': 230,
            'raw_y_max': 250,
            'ocr_text': 'Digitized by eGangotri Archive',
            'ocr_conf': 0.95,
            'latin_count': 25,
            'latin_ratio': 1.0,
            'deva_ratio': 0.0,
            'gap_above': 50,
            'gap_below': 50,
            'rel_pos': 0.5,
            'height': 20,
        }
        ms_line_1 = {
            'bbox': {'x_min': 100, 'y_min': 50, 'x_max': 700, 'y_max': 180},
            'raw_y_min': 50,
            'raw_y_max': 180,
            'ocr_text': 'श्रीगणेशाय नमः',
            'ocr_conf': 0.90,
            'latin_count': 0,
            'latin_ratio': 0.0,
            'deva_ratio': 1.0,
            'gap_above': 10,
            'gap_below': 50,
            'rel_pos': 0.2,
            'height': 130,
        }
        ms_line_2 = {
            'bbox': {'x_min': 100, 'y_min': 300, 'x_max': 700, 'y_max': 430},
            'raw_y_min': 300,
            'raw_y_max': 430,
            'ocr_text': 'अथ प्रथमोऽध्यायः',
            'ocr_conf': 0.90,
            'latin_count': 0,
            'latin_ratio': 0.0,
            'deva_ratio': 1.0,
            'gap_above': 50,
            'gap_below': 10,
            'rel_pos': 0.8,
            'height': 130,
        }

        # Mock extract_lines to return this 3-line sequence
        refiner = SemanticRefiner()
        refiner.line_analyzer.extract_lines = lambda im, b: [ms_line_1, filler_line, ms_line_2]

        refined = refiner.refine_detections(img, [candidate])
        main_boxes = [d for d in refined if d['class'] == 'main_text']
        filler_boxes = [d for d in refined if d['class'] == 'filler']

        self.assertEqual(len(filler_boxes), 1)
        self.assertEqual(len(main_boxes), 2)
        # Verify neither main_text overlaps the filler box
        fb = filler_boxes[0]['bbox']
        for mb in main_boxes:
            b = mb['bbox']
            # Either completely above or completely below
            self.assertTrue(b['y_max'] <= fb['y_min'] or b['y_min'] >= fb['y_max'])

    # 4. Absent classes & JSON requirements
    def test_json_output_strict_classes_and_no_hallucinations(self):
        """JSON output must strictly permit only the 5 allowed classes and drop any invalid class."""
        dets = [
            {'class': 'main_text', 'confidence': 0.91, 'bbox': {'x_min': 10, 'y_min': 20, 'x_max': 100, 'y_max': 200}},
            {'class': 'filler', 'confidence': 0.88, 'bbox': {'x_min': 10, 'y_min': 210, 'x_max': 100, 'y_max': 230}},
            {'class': 'paragraph', 'confidence': 0.50, 'bbox': {'x_min': 0, 'y_min': 0, 'x_max': 10, 'y_max': 10}},  # should be filtered
        ]
        with tempfile.TemporaryDirectory() as td:
            save_json_metadata(dets, 'test.png', (200, 300), td, 0.1)
            jp = os.path.join(td, 'metadata', 'test_metadata.json')
            with open(jp, 'r') as f:
                data = json.load(f)

            allowed = {'header', 'footer', 'main_text', 'side_text', 'filler'}
            labels = [d['label'] for d in data['detections']]
            self.assertEqual(len(labels), 2)
            for label in labels:
                self.assertIn(label, allowed)
            self.assertNotIn('paragraph', labels)
            self.assertTrue(all('class' not in d for d in data['detections']))
            # Absent classes like 'header', 'footer', 'side_text' are NOT hallucinated
            self.assertNotIn('header', labels)
            self.assertNotIn('footer', labels)
            self.assertNotIn('side_text', labels)

    # 5. Bounding box boundary clipping
    def test_bbox_clipping(self):
        """All bounding boxes must be clamped to image dimensions [0, w] and [0, h]."""
        dets = [
            {'class': 'main_text', 'bbox': {'x_min': -20, 'y_min': -5, 'x_max': 850, 'y_max': 410}},
        ]
        clipped = clip_boxes(dets, w=800, h=400)
        b = clipped[0]['bbox']
        self.assertEqual(b['x_min'], 0)
        self.assertEqual(b['y_min'], 0)
        self.assertEqual(b['x_max'], 800)
        self.assertEqual(b['y_max'], 400)

    # 6. OCR failure robustness & semantic metadata tests
    def test_ocr_failure_graceful_handling(self):
        """Pipeline must NEVER crash when the OCR recognizer throws arbitrary exceptions."""
        class CrashingRecognizer:
            def predict(self, *args, **kwargs):
                raise RuntimeError("Simulated OCR engine C++ / oneDNN internal crash")

        refiner = SemanticRefiner(recognizer=CrashingRecognizer())
        img = np.full((300, 500, 3), 200, dtype=np.uint8)
        cand = {
            'class': 'main_text',
            'confidence': 0.85,
            'bbox': {'x_min': 50, 'y_min': 50, 'x_max': 450, 'y_max': 250},
        }

        # Must not raise an exception; must gracefully return valid detections
        try:
            refined = refiner.refine_detections(img, [cand])
            self.assertIsInstance(refined, list)
            self.assertGreaterEqual(len(refined), 1)
        except Exception as e:
            self.fail(f"refine_detections raised {type(e).__name__}: {e}")

    def test_ocr_metadata_enrichment_in_json(self):
        """JSON output must include ocr_text, ocr_confidence, devanagari_ratio, latin_ratio when present."""
        dets = [
            {
                'class': 'header',
                'confidence': 0.95,
                'bbox': {'x_min': 100, 'y_min': 50, 'x_max': 700, 'y_max': 120},
                'ocr_text': '॥ श्रीमद्भगवद्गीता ॥',
                'ocr_conf': 0.97,
                'deva_ratio': 1.0,
                'latin_ratio': 0.0,
            },
            {
                'class': 'filler',
                'confidence': 0.92,
                'bbox': {'x_min': 100, 'y_min': 900, 'x_max': 700, 'y_max': 950},
                'ocr_text': 'Source: Bhagavad Gita translated by Swami Sivananda',
                'ocr_conf': 0.96,
                'deva_ratio': 0.0,
                'latin_ratio': 0.92,
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            save_json_metadata(dets, 'test_ocr.png', (800, 1000), td, 0.5)
            jp = os.path.join(td, 'metadata', 'test_ocr_metadata.json')
            with open(jp, 'r', encoding='utf-8') as f:
                data = json.load(f)

            d0 = data['detections'][0]
            self.assertEqual(d0['label'], 'header')
            self.assertEqual(d0['ocr_text'], '॥ श्रीमद्भगवद्गीता ॥')
            self.assertEqual(d0['devanagari_ratio'], 1.0)
            self.assertEqual(d0['latin_ratio'], 0.0)

            d1 = data['detections'][1]
            self.assertEqual(d1['label'], 'filler')
            self.assertEqual(d1['devanagari_ratio'], 0.0)
            self.assertGreater(d1['latin_ratio'], 0.80)

    # 4. Regression tests for specific detection failures
    def test_english_attribution_classified_as_filler(self):
        """Regression test: English/Latin attribution text must be classified as filler even when OCR is weak."""
        # Case A: Low OCR confidence attribution text with keywords
        line_weak_ocr = {
            'ocr_text': 'Digitized by eGangotri, MIDF Funding, Public Domain',
            'ocr_conf': 0.22,
            'latin_count': 35,
            'latin_ratio': 0.85,
            'deva_ratio': 0.0,
            'gap_above': 20,
            'gap_below': 10,
            'rel_pos': 0.95,
            'height': 15,
        }
        score = self.refiner.calculate_filler_score(line_weak_ocr)
        self.assertGreaterEqual(score, self.refiner.score_threshold)

        # Case B: Marginal Latin attribution text with weak OCR
        line_marginal_latin = {
            'ocr_text': 'acc. no. 1420/beng region',
            'ocr_conf': 0.25,
            'latin_count': 18,
            'latin_ratio': 0.80,
            'deva_ratio': 0.0,
            'gap_above': 30,
            'gap_below': 5,
            'rel_pos': 0.96,
            'height': 14,
        }
        score2 = self.refiner.calculate_filler_score(line_marginal_latin)
        self.assertGreaterEqual(score2, self.refiner.score_threshold)

    def test_blank_predicted_text_box_rejected(self):
        """Regression test: candidate text box on blank margins/background must be rejected."""
        # Create blank image with no ink
        blank_img = np.full((500, 500, 3), 240, dtype=np.uint8)
        # Candidate text box predicted in blank margin area
        cand = {
            'bbox': {'x_min': 50, 'y_min': 20, 'x_max': 450, 'y_max': 120},
            'class': 'main_text',
            'confidence': 0.82,
        }
        refined = self.refiner.refine_detections(blank_img, [cand])
        # Box has no ink / no text evidence -> must be rejected
        self.assertEqual(len(refined), 0)

    def test_filler_overlapping_inside_main_text_corrected_or_split(self):
        """Regression test: filler inside or overlapping main_text causes main_text to be corrected/split."""
        from src.postprocessor import resolve_conflicts
        # main_text covering [100, 100, 900, 900]
        main_cand = {
            'class': 'main_text',
            'confidence': 0.90,
            'bbox': {'x_min': 100, 'y_min': 100, 'x_max': 900, 'y_max': 900},
        }
        # filler region inside the middle of main_text [150, 450, 850, 520]
        filler_cand = {
            'class': 'filler',
            'confidence': 0.92,
            'bbox': {'x_min': 150, 'y_min': 450, 'x_max': 850, 'y_max': 520},
        }

        resolved = resolve_conflicts([main_cand, filler_cand], w=1000, h=1000)
        mains = [d for d in resolved if d['class'] == 'main_text']
        fillers = [d for d in resolved if d['class'] == 'filler']

        # Filler must be retained
        self.assertEqual(len(fillers), 1)
        # Main text must be split into top and bottom regions
        self.assertEqual(len(mains), 2)

        # Ensure no main_text region contains or overlaps the filler box
        fb = fillers[0]['bbox']
        for m in mains:
            mb = m['bbox']
            # Either completely above filler or completely below filler
            no_overlap = (mb['y_max'] < fb['y_min']) or (mb['y_min'] > fb['y_max'])
            self.assertTrue(no_overlap, f"main_text {mb} overlaps filler {fb}")

    def _owned_line(self, y, x1=180, x2=820, text='', latin=0, deva=20):
        """Create a deterministic OCR-line fixture without model inference."""
        return {
            'bbox': {'x_min': x1, 'y_min': y, 'x_max': x2, 'y_max': y + 18},
            'height': 18, 'width': x2 - x1, 'ocr_text': text, 'ocr_conf': 0.96,
            'latin_count': latin, 'devanagari_count': deva,
            'total_chars': latin + deva, 'latin_ratio': latin / max(1, latin + deva),
            'deva_ratio': deva / max(1, latin + deva),
        }

    def test_page_ownership_keeps_bottom_manuscript_line(self):
        """A body-continuous lower line is not filler merely because it is low on the page."""
        image = np.full((800, 1000, 3), 220, dtype=np.uint8)
        lines = [self._owned_line(y) for y in (560, 590, 620, 650)]
        featured = self.refiner.attach_page_features(image, lines, {'metrics': {'h_line': 18}}, [])
        owned = self.refiner.assign_ownership(featured, {'metrics': {'h_line': 18}}, image.shape[:2])
        self.assertEqual([line['owner'] for line in owned], ['main_text'] * 4)

    def test_page_ownership_splits_english_attribution_without_losing_last_body_line(self):
        """Only the attribution line becomes filler; the final manuscript line remains body."""
        image = np.full((800, 1000, 3), 220, dtype=np.uint8)
        cv2.putText(image, 'CSU-Sringeri Campus Rare Book and Manuscript Collection',
                    (240, 706), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
        body = [self._owned_line(y) for y in (560, 590, 620, 650)]
        attribution = self._owned_line(
            690, 240, 780,
            'CSU-Sringeri Campus Rare Book and Manuscript Collection', latin=52, deva=0,
        )
        featured = self.refiner.attach_page_features(image, body + [attribution], {'metrics': {'h_line': 18}}, [])
        owned = self.refiner.assign_ownership(featured, {'metrics': {'h_line': 18}}, image.shape[:2])
        self.assertEqual(owned[-1]['owner'], 'filler')
        self.assertEqual(owned[-2]['owner'], 'main_text')
        regions = self.refiner.rebuild_owned_regions(image, owned)
        main = next(region for region in regions if region['class'] == 'main_text')
        filler = next(region for region in regions if region['class'] == 'filler')
        self.assertGreaterEqual(main['bbox']['y_max'], 668)
        self.assertLess(main['bbox']['y_max'], filler['bbox']['y_min'])

    def test_page_ownership_preserves_independent_footer_and_side_flow(self):
        """Non-body flows get semantic owners from geometry relative to the body flow."""
        image = np.full((1000, 1200, 3), 220, dtype=np.uint8)
        body = [self._owned_line(y, 300, 900) for y in (250, 285, 320, 355)]
        left_note = [self._owned_line(y, 40, 190) for y in (270, 305)]
        footer = self._owned_line(880, 450, 650)
        featured = self.refiner.attach_page_features(image, body + left_note + [footer], {'metrics': {'h_line': 18}}, [])
        owned = self.refiner.assign_ownership(featured, {'metrics': {'h_line': 18}}, image.shape[:2])
        owners = {(line['bbox']['x_min'], line['bbox']['y_min']): line['owner'] for line in owned}
        self.assertEqual(owners[(40, 270)], 'side_text')
        self.assertEqual(owners[(450, 880)], 'footer')

    def test_page_structure_owns_interior_sides_header_footers_and_source(self):
        """Flow ownership is structural, so sides need not touch an outer page margin."""
        image = np.full((900, 1200, 3), 225, dtype=np.uint8)
        header = [self._owned_line(90, 420, 780)]
        left = [self._owned_line(y, 150, 285) for y in (220, 260, 300, 340, 380)]
        body = [self._owned_line(y, 350, 850) for y in (210, 250, 290, 330, 370, 410)]
        right = [self._owned_line(y, 910, 1050) for y in (220, 260, 300, 340, 380)]
        footer_left = [self._owned_line(740, 260, 350)]
        footer_right = [self._owned_line(740, 850, 940)]
        source = [self._owned_line(
            830, 260, 940, 'Source: Bhagavad Gita translated by Swami Sivananda', latin=48, deva=0
        )]
        cv2.putText(image, 'Source: Bhagavad Gita translated by Swami Sivananda',
                    (260, 844), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
        named_flows = {
            'header': header, 'left': left, 'body': body, 'right': right,
            'footer-left': footer_left, 'footer-right': footer_right, 'source': source,
        }
        lines = []
        for flow_id, flow_lines in named_flows.items():
            for line in flow_lines:
                line['flow_id'] = flow_id
                lines.append(line)
        structure = {
            'metrics': {'h_line': 18, 'g_vert': 22},
            'content_frame': {'x_min': 100, 'y_min': 60, 'x_max': 1100, 'y_max': 790},
            'separators': {'horizontal': [165, 700], 'vertical': [320, 875]},
        }
        featured = self.refiner.attach_page_features(image, lines, structure, [])
        owned = self.refiner.assign_ownership(featured, structure, image.shape[:2])
        owners = {flow_id: {line['owner'] for line in owned if line['flow_id'] == flow_id}
                  for flow_id in named_flows}

        self.assertEqual(owners['header'], {'header'})
        self.assertEqual(owners['left'], {'side_text'})
        self.assertEqual(owners['body'], {'main_text'})
        self.assertEqual(owners['right'], {'side_text'})
        self.assertEqual(owners['footer-left'], {'footer'})
        self.assertEqual(owners['footer-right'], {'footer'})
        self.assertEqual(owners['source'], {'filler'})

        regions = self.refiner.rebuild_owned_regions(image, owned)
        self.assertEqual(len([r for r in regions if r['class'] == 'side_text']), 2)
        self.assertEqual(len([r for r in regions if r['class'] == 'footer']), 2)
        main = next(r for r in regions if r['class'] == 'main_text')
        self.assertGreater(main['bbox']['x_min'], 320)
        self.assertLess(main['bbox']['x_max'], 875)

    def test_isolated_symbol_motif_is_filler_not_a_header(self):
        """Non-text decorative motifs are filler even when they precede a title."""
        image = np.full((900, 1200, 3), 225, dtype=np.uint8)
        title = self._owned_line(90, 420, 780)
        body = [self._owned_line(y, 350, 850) for y in (210, 250, 290, 330)]
        left_motif = self._owned_line(105, 150, 205, '*', latin=0, deva=0)
        right_motif = self._owned_line(105, 840, 895, '*', latin=0, deva=0)
        for flow_id, flow_lines in {
            'title': [title], 'body': body,
            'left-motif': [left_motif], 'right-motif': [right_motif],
        }.items():
            for line in flow_lines:
                line['flow_id'] = flow_id
        lines = [title, *body, left_motif, right_motif]
        structure = {
            'metrics': {'h_line': 18, 'g_vert': 22},
            'content_frame': {'x_min': 100, 'y_min': 60, 'x_max': 1100, 'y_max': 700},
            'separators': {'horizontal': [165], 'vertical': []},
        }

        featured = self.refiner.attach_page_features(image, lines, structure, [])
        owned = self.refiner.assign_ownership(featured, structure, image.shape[:2])
        owners = {line['flow_id']: line['owner'] for line in owned}

        self.assertEqual(owners['title'], 'header')
        self.assertEqual(owners['left-motif'], 'filler')
        self.assertEqual(owners['right-motif'], 'filler')
        filler_groups = self.refiner._group_owned_lines(owned, 'filler')
        self.assertEqual(len(filler_groups), 2)

    def test_overlapping_body_fragments_rebuild_as_one_flow(self):
        """Duplicate OCR fragments cannot become overlapping main-text regions."""
        image = np.full((900, 1200, 3), 225, dtype=np.uint8)
        upper = [self._owned_line(y, 220, 900) for y in (200, 240, 280, 320, 360, 400, 440)]
        # This smaller nested fragment falls below the primary-body score
        # threshold, then reaches the default main role; it must still merge.
        lower = [self._owned_line(y, 230, 890) for y in (360, 400)]
        for flow_id, flow_lines in {'upper': upper, 'lower': lower}.items():
            for line in flow_lines:
                line['flow_id'] = flow_id
        structure = {'metrics': {'h_line': 18, 'g_vert': 22}, 'separators': {'horizontal': [], 'vertical': []}}

        featured = self.refiner.attach_page_features(image, [*upper, *lower], structure, [])
        owned = self.refiner.assign_ownership(featured, structure, image.shape[:2])
        mains = self.refiner._group_owned_lines(owned, 'main_text')

        self.assertEqual(len(mains), 1)


if __name__ == '__main__':
    unittest.main()
