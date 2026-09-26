"""Unit tests for the detector (no model required — tests the fallback path)."""

import os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.detector import ManuscriptDetector


class TestDetector(unittest.TestCase):
    """Validate detector logic using OpenCV fallback."""

    def setUp(self):
        self.detector = ManuscriptDetector.__new__(ManuscriptDetector)
        self.detector.config = {}
        self.detector.class_mapping = {}
        self.detector.model = None
        self.detector.model_available = False
        self.detector.header_zone = 0.18
        self.detector.footer_zone = 0.82
        self.detector.side_margin = 0.15
        self.detector.direct_map = {'text': 'main_text'}
        self.detector.candidate_map = {}
        self.detector.ignore_labels = set()
        self.detector.recognizer = None

    def _page(self) -> np.ndarray:
        """Create a synthetic manuscript page: light center on dark bg."""
        img = np.full((400, 600, 3), 40, dtype=np.uint8)
        img[50:350, 80:520] = 220  # light manuscript area
        cv2 = __import__('cv2')
        cv2.putText(img, "Header text", (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        cv2.putText(img, "Main body text here", (150, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
        return img

    def test_manuscript_region(self):
        img = self._page()
        ms = self.detector._find_manuscript_region(img)
        self.assertIn('x_min', ms)
        self.assertIn('y_min', ms)
        self.assertTrue(ms['x_max'] > ms['x_min'])
        self.assertTrue(ms['y_max'] > ms['y_min'])

    def test_no_fallback_when_model_unavailable(self):
        """When model is not available, detect() returns empty without fallback."""
        img = self._page()
        from src.semantic_refiner import SemanticRefiner
        self.detector.refiner = SemanticRefiner(recognizer=None, config={})
        dets = self.detector.detect('dummy.png', img)
        self.assertEqual(dets, [])

    def test_spatial_classify(self):
        bbox_top = {'x_min': 100, 'y_min': 10, 'x_max': 500, 'y_max': 50}
        ms = {'x_min': 0, 'y_min': 0, 'x_max': 600, 'y_max': 400}
        cls = self.detector._spatial_classify(bbox_top, 600, 400, ms)
        self.assertEqual(cls, 'header')

    def test_class_mapping_v2_taxonomy_and_target_classes(self):
        """Audit test: class_mapping.yaml must use strictly official PP-DocLayoutV2 labels."""
        from src.utils import load_class_mapping
        cm = load_class_mapping()

        official_v2_labels = {
            'abstract', 'algorithm', 'aside_text', 'chart', 'content',
            'display_formula', 'doc_title', 'figure_title', 'footer',
            'footer_image', 'footnote', 'formula_number', 'header',
            'header_image', 'image', 'inline_formula', 'number',
            'paragraph_title', 'reference', 'reference_content', 'seal',
            'table', 'text', 'vertical_text', 'vision_footnote'
        }
        allowed_targets = {'header', 'main_text', 'footer', 'side_text', 'filler'}

        # Verify target classes
        self.assertEqual(set(cm.get('target_classes', [])), allowed_targets)

        # Verify direct mappings
        for k, v in cm.get('direct_mappings', {}).items():
            self.assertIn(k, official_v2_labels, f"Direct mapping key '{k}' is not an official V2 label")
            self.assertIn(v, allowed_targets, f"Direct mapping target '{v}' is not an allowed assignment class")

        # Verify candidate mappings
        for k, v in cm.get('candidate_mappings', {}).items():
            self.assertIn(k, official_v2_labels, f"Candidate mapping key '{k}' is not an official V2 label")
            self.assertIn(v, allowed_targets, f"Candidate mapping target '{v}' is not an allowed assignment class")

        # Verify ignore labels
        for k in cm.get('ignore_labels', []):
            self.assertIn(k, official_v2_labels, f"Ignore label '{k}' is not an official V2 label")

        # Confirm non-existent labels are absent
        forbidden = {'paragraph', 'title', 'handwriting', 'figure', 'stamp', 'aside', 'equation', 'table_caption'}
        for f in forbidden:
            self.assertNotIn(f, cm.get('direct_mappings', {}))
            self.assertNotIn(f, cm.get('candidate_mappings', {}))
            self.assertNotIn(f, cm.get('ignore_labels', []))

    def test_validate_candidate_geometric_rules(self):
        """Candidate mappings must not be unconditional; verified via geometry/context."""
        ms = {'x_min': 0, 'y_min': 0, 'x_max': 1000, 'y_max': 1000}

        # 1. aside_text: side_text only when in margin
        marginal_det = {'bbox': {'x_min': 10, 'y_min': 200, 'x_max': 120, 'y_max': 800}, 'raw_label': 'aside_text'}
        self.assertEqual(self.detector._validate_candidate(marginal_det, 'side_text', 1000, 1000, ms), 'side_text')

        interior_det = {'bbox': {'x_min': 300, 'y_min': 200, 'x_max': 700, 'y_max': 800}, 'raw_label': 'aside_text'}
        self.assertEqual(self.detector._validate_candidate(interior_det, 'side_text', 1000, 1000, ms), 'main_text')

        # 2. paragraph_title: header only when positioned as header content in top band
        top_title = {'bbox': {'x_min': 200, 'y_min': 20, 'x_max': 800, 'y_max': 80}, 'raw_label': 'paragraph_title'}
        self.assertEqual(self.detector._validate_candidate(top_title, 'header', 1000, 1000, ms), 'header')

        body_title = {'bbox': {'x_min': 200, 'y_min': 400, 'x_max': 800, 'y_max': 450}, 'raw_label': 'paragraph_title'}
        self.assertEqual(self.detector._validate_candidate(body_title, 'header', 1000, 1000, ms), 'main_text')

        # 3. image/seal: filler only when consistent with assignment definition (rejects full-page false positives)
        huge_image = {'bbox': {'x_min': 50, 'y_min': 50, 'x_max': 950, 'y_max': 950}, 'raw_label': 'image'}
        self.assertIsNone(self.detector._validate_candidate(huge_image, 'filler', 1000, 1000, ms))

        valid_seal = {'bbox': {'x_min': 800, 'y_min': 800, 'x_max': 950, 'y_max': 950}, 'raw_label': 'seal'}
        self.assertEqual(self.detector._validate_candidate(valid_seal, 'filler', 1000, 1000, ms), 'filler')


if __name__ == '__main__':
    unittest.main()
