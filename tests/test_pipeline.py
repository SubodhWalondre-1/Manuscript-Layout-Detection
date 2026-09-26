"""End-to-end integration test for the full pipeline."""

import os, sys, unittest, tempfile, json
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_config, load_class_mapping
from src.detector import ManuscriptDetector
from src.postprocessor import postprocess, tighten_bbox_to_content, suppress_container_boxes
from src.visualizer import draw_detections
from src.output_handler import save_annotated_image, save_json_metadata


class TestPipeline(unittest.TestCase):

    def _synthetic_image(self, path: str) -> str:
        img = np.full((400, 600, 3), 50, dtype=np.uint8)
        img[60:340, 90:510] = 210
        cv2.putText(img, "Title line", (120, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        cv2.putText(img, "Body content", (140, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        cv2.putText(img, "Attribution", (200, 370),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        cv2.imwrite(path, img)
        return path

    def test_full_pipeline(self):
        """Synthetic image through detect → postprocess → visualise → save."""
        cfg = load_config()
        cm = load_class_mapping()

        det = ManuscriptDetector.__new__(ManuscriptDetector)
        det.config = cfg
        det.class_mapping = cm
        class MockModel:
            def predict(self, path, **kwargs):
                return [{
                    'boxes': [
                        {'coordinate': [140, 150, 500, 250], 'label': 'paragraph', 'score': 0.90},
                    ]
                }]

        det.model = MockModel()
        det.model_available = True
        det.header_zone = 0.18
        det.footer_zone = 0.82
        det.side_margin = 0.15
        det.direct_map = cm.get('direct_mappings', {})
        det.candidate_map = cm.get('candidate_mappings', {})
        det.ignore_labels = set(cm.get('ignore_labels', []))
        det.recognizer = None
        from src.semantic_refiner import SemanticRefiner
        det.refiner = SemanticRefiner(recognizer=None, config=cfg)

        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, 'test.png')
            self._synthetic_image(img_path)
            original = cv2.imread(img_path)

            dets = det.detect(img_path, original, 0.15)
            h, w = original.shape[:2]
            dets = postprocess(dets, w, h, 0.15, cfg, image=original)

            for d in dets:
                b = d['bbox']
                self.assertGreaterEqual(b['x_min'], 0)
                self.assertGreaterEqual(b['y_min'], 0)
                self.assertLessEqual(b['x_max'], w)
                self.assertLessEqual(b['y_max'], h)

            ann = draw_detections(original, dets, cfg)
            self.assertEqual(ann.shape, original.shape)

            save_annotated_image(ann, td, 'test.png')
            self.assertTrue(os.path.isfile(
                os.path.join(td, 'annotated', 'test_annotated.png')
            ))

            save_json_metadata(dets, 'test.png', (w, h), td, 0.5)
            jp = os.path.join(td, 'metadata', 'test_metadata.json')
            self.assertTrue(os.path.isfile(jp))
            with open(jp) as f:
                meta = json.load(f)
            self.assertEqual(meta['image_name'], 'test.png')
            self.assertIn('detections', meta)

    def test_tighten_bbox_to_content(self):
        """Verify that padding around text is trimmed by tighten_bbox_to_content."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        # Put small text in the center
        cv2.putText(img, "TEST", (60, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Loose initial box with 20px padding on every side
        loose_bbox = {'x_min': 20, 'y_min': 10, 'x_max': 180, 'y_max': 90}
        tight_bbox, density = tighten_bbox_to_content(img, loose_bbox, pad=2)
        # Bounding box should be tightened
        self.assertGreater(tight_bbox['x_min'], loose_bbox['x_min'])
        self.assertLess(tight_bbox['x_max'], loose_bbox['x_max'])
        self.assertGreater(density, 0.10)

    def test_suppress_container_boxes(self):
        """Verify that large parent filler boxes are suppressed when containing small tight filler boxes."""
        dets = [
            {'class': 'filler', 'bbox': {'x_min': 0, 'y_min': 300, 'x_max': 1000, 'y_max': 400}, 'confidence': 0.90},
            {'class': 'filler', 'bbox': {'x_min': 250, 'y_min': 340, 'x_max': 700, 'y_max': 370}, 'confidence': 0.88},
            {'class': 'main_text', 'bbox': {'x_min': 100, 'y_min': 50, 'x_max': 900, 'y_max': 280}, 'confidence': 0.95},
        ]
        filtered = suppress_container_boxes(dets)
        # The large filler box should be suppressed, keeping only the tight filler and main_text
        self.assertEqual(len(filtered), 2)
        classes = [d['class'] for d in filtered]
        self.assertIn('filler', classes)
        self.assertIn('main_text', classes)
        filler_box = next(d for d in filtered if d['class'] == 'filler')['bbox']
        self.assertEqual(filler_box['x_min'], 250)

    def test_second_stage_main_text_refinement(self):
        """Verify that an absorbed English line is split out of main_text into filler."""
        det = ManuscriptDetector.__new__(ManuscriptDetector)
        det.config = {}
        det.class_mapping = {}
        det.recognizer = None  # Use heuristic / mock

        # Mock recognizer for testing: only returns attribution for bottom line
        class MockRecognizer:
            def predict(self, crop):
                # When crop is from bottom strip (small dark background)
                if np.mean(crop) < 100:
                    return [{'rec_text': 'CSU-Sringeri Campus Rare Book and Manuscript Collection, Sringeri',
                             'rec_score': 0.98}]
                return [{'rec_text': '', 'rec_score': 0.0}]

        det.recognizer = MockRecognizer()

        # Create an image with a manuscript block and an attribution line at bottom
        img = np.full((400, 800, 3), 40, dtype=np.uint8)
        # Manuscript body: y=50..300
        img[50:300, 100:700] = 220
        cv2.putText(img, "Manuscript writing body line", (120, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
        # Attribution text line at y=365..385
        cv2.putText(img, "CSU-Sringeri Campus Rare Book and Manuscript Collection, Sringeri",
                    (150, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)

        # Candidate main_text box that absorbed the English line
        candidate_det = {
            'bbox': {'x_min': 100, 'y_min': 50, 'x_max': 700, 'y_max': 390},
            'class': 'main_text',
            'confidence': 0.85
        }

        m_list, f_list = det._inspect_and_split_main_text(img, candidate_det)

        # Should split into 1 refined main_text and 1 filler
        self.assertEqual(len(m_list), 1)
        self.assertEqual(len(f_list), 1)
        self.assertEqual(m_list[0]['class'], 'main_text')
        self.assertEqual(f_list[0]['class'], 'filler')

        # Refined main_text must NOT include the bottom English line
        self.assertLess(m_list[0]['bbox']['y_max'], 350)
        # Filler line must be around the bottom text
        self.assertGreaterEqual(f_list[0]['bbox']['y_min'], 350)

    def test_header_area_filtering(self):
        """Verify that compact header boxes (folio numbers) are preserved while noise is filtered."""
        from src.postprocessor import filter_area
        img_area = 1000 * 1000  # 1,000,000 px
        # Default min_ratio=0.003 -> 3000 px; min_header_ratio=0.002 -> 2000 px
        dets = [
            {'class': 'header', 'bbox': {'x_min': 50, 'y_min': 50, 'x_max': 150, 'y_max': 75}, 'confidence': 0.85},   # area = 2500 px
            {'class': 'header', 'bbox': {'x_min': 50, 'y_min': 50, 'x_max': 80, 'y_max': 60}, 'confidence': 0.20},    # area = 300 px (noise)
            {'class': 'main_text', 'bbox': {'x_min': 50, 'y_min': 50, 'x_max': 150, 'y_max': 75}, 'confidence': 0.85},# area = 2500 px (< 3000 px)
        ]
        kept = filter_area(dets, img_area, min_ratio=0.003, min_header_ratio=0.002)
        # Only the compact valid header (2500 px >= 2000 px) should be kept
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]['class'], 'header')
        self.assertEqual(kept[0]['bbox']['x_max'], 150)

    def test_image_19_end_to_end_layout(self):
        """Verify full tripartite commentary layout decomposition into all 5 target classes."""
        cfg = load_config()
        cm = load_class_mapping()

        img_path = './data/test_images/image_19.png'
        if not os.path.exists(img_path):
            self.skipTest("image_19.png not present in data/test_images/")

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        det = ManuscriptDetector.__new__(ManuscriptDetector)
        det.config = cfg
        det.class_mapping = cm
        det.device = 'cpu'
        det.header_zone = 0.18
        det.footer_zone = 0.82
        det.side_margin = 0.15
        det.direct_map = cm.get('direct_mappings', {})
        det.candidate_map = cm.get('candidate_mappings', {})
        det.ignore_labels = set(cm.get('ignore_labels', []))
        det.recognizer = None
        from src.semantic_refiner import SemanticRefiner
        det.refiner = SemanticRefiner(recognizer=None, config=cfg)

        # Mock LayoutDetection returning macro table container and footnote attribution
        class MockModel:
            def predict(self, path, **kwargs):
                return [{
                    'boxes': [
                        {'coordinate': [31, 78, 994, 1426], 'label': 'table', 'score': 0.968},
                        {'coordinate': [145, 1447, 890, 1483], 'label': 'vision_footnote', 'score': 0.463},
                    ]
                }]

        det.model = MockModel()
        det.model_available = True

        dets = det.detect(img_path, img, 0.15)
        dets = postprocess(dets, w, h, 0.15, cfg, image=img)

        classes = {d['class'] for d in dets}
        # Must contain all 5 target classes!
        self.assertIn('header', classes)
        self.assertIn('main_text', classes)
        self.assertIn('side_text', classes)
        self.assertIn('footer', classes)
        self.assertIn('filler', classes)

        # Verify spatial hierarchy
        hdr = next(d for d in dets if d['class'] == 'header')['bbox']
        main = next(d for d in dets if d['class'] == 'main_text')['bbox']
        ftr = next(d for d in dets if d['class'] == 'footer')['bbox']
        fil = next(d for d in dets if d['class'] == 'filler')['bbox']
        sides = [d['bbox'] for d in dets if d['class'] == 'side_text']

        # Header above main text
        self.assertLessEqual(hdr['y_max'], main['y_min'] + 10)
        # Main text above footer
        self.assertLessEqual(main['y_max'], ftr['y_min'] + 10)
        # Footer above bottom filler
        self.assertLessEqual(ftr['y_max'], fil['y_min'] + 10)
        # Exactly 2 side text columns flanking main text
        self.assertEqual(len(sides), 2)
        left_side = min(sides, key=lambda b: b['x_min'])
        right_side = max(sides, key=lambda b: b['x_min'])
        self.assertLessEqual(left_side['x_max'], main['x_min'] + 10)
        self.assertGreaterEqual(right_side['x_min'], main['x_max'] - 10)


if __name__ == '__main__':
    unittest.main()
