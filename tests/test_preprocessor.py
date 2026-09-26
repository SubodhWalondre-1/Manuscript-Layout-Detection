"""Unit tests for the preprocessing module."""

import os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessor import (
    deskew, enhance, letterbox_resize, fix_orientation,
    restore_bbox, restore_detections, preprocess, load_image,
)
from src.utils import resolve_device


class TestPreprocessor(unittest.TestCase):
    """Validate preprocessing functions and coordinate restoration."""

    def _blank(self, w: int = 200, h: int = 100) -> np.ndarray:
        return np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)

    def test_deskew_returns_same_shape(self):
        img = self._blank()
        out = deskew(img)
        self.assertEqual(img.shape, out.shape)

    def test_enhance_returns_same_shape(self):
        img = self._blank()
        out = enhance(img)
        self.assertEqual(img.shape, out.shape)

    def test_letterbox_returns_square(self):
        img = self._blank(300, 100)
        out, info = letterbox_resize(img, 256)
        self.assertEqual(out.shape[:2], (256, 256))
        self.assertIn('scale', info)
        self.assertIn('pad_left', info)

    def test_letterbox_preserves_info(self):
        img = self._blank(640, 480)
        _, info = letterbox_resize(img, 512)
        self.assertEqual(info['original_width'], 640)
        self.assertEqual(info['original_height'], 480)

    def test_restore_bbox_exact_mapping(self):
        """Verify that letterbox coordinates restore accurately to original image space."""
        orig_w, orig_h = 1000, 500
        img = self._blank(orig_w, orig_h)
        target_size = 500
        padded, info = letterbox_resize(img, target_size)

        # A region in the original image: (100, 50, 400, 200)
        # In letterbox space: scale = 500/1000 = 0.5, pad_top = (500 - 250) // 2 = 125, pad_left = 0
        scale = info['scale']
        pad_l, pad_t = info['pad_left'], info['pad_top']
        lb_box = {
            'x_min': int(100 * scale + pad_l),
            'y_min': int(50 * scale + pad_t),
            'x_max': int(400 * scale + pad_l),
            'y_max': int(200 * scale + pad_t),
        }

        restored = restore_bbox(lb_box, info)
        self.assertAlmostEqual(restored['x_min'], 100, delta=2)
        self.assertAlmostEqual(restored['y_min'], 50, delta=2)
        self.assertAlmostEqual(restored['x_max'], 400, delta=2)
        self.assertAlmostEqual(restored['y_max'], 200, delta=2)

    def test_restore_detections_batch(self):
        """Verify restore_detections restores all bboxes in list."""
        transform_info = {
            'scale': 0.5,
            'pad_left': 10,
            'pad_top': 20,
            'original_width': 800,
            'original_height': 600,
        }
        dets = [
            {'class': 'header', 'bbox': {'x_min': 60, 'y_min': 70, 'x_max': 210, 'y_max': 120}},
            {'class': 'main_text', 'bbox': {'x_min': 110, 'y_min': 170, 'x_max': 310, 'y_max': 270}},
        ]
        restored = restore_detections(dets, transform_info)
        self.assertEqual(len(restored), 2)
        # (60 - 10) / 0.5 = 100, (70 - 20) / 0.5 = 100
        self.assertEqual(restored[0]['bbox']['x_min'], 100)
        self.assertEqual(restored[0]['bbox']['y_min'], 100)

    def test_resolve_device(self):
        """Verify device string resolution behavior."""
        self.assertEqual(resolve_device('cpu'), 'cpu')
        # On this CPU-only environment, auto and gpu resolve safely
        self.assertIn(resolve_device('auto'), ['cpu', 'gpu'])
        self.assertIn(resolve_device('gpu'), ['cpu', 'gpu'])

    def test_input_immutability(self):
        """Verify that loading, EXIF check, and preprocess never alter the disk file."""
        import tempfile
        import hashlib
        import cv2

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create a test image
            arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(tmp_path, arr)

            with open(tmp_path, 'rb') as f:
                orig_hash = hashlib.sha256(f.read()).hexdigest()

            # Execute load and preprocess
            loaded = load_image(tmp_path)
            fixed = fix_orientation(loaded, tmp_path)
            proc, orig, trans = preprocess(tmp_path)

            with open(tmp_path, 'rb') as f:
                post_hash = hashlib.sha256(f.read()).hexdigest()

            self.assertEqual(orig_hash, post_hash, "Input file was modified on disk!")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == '__main__':
    unittest.main()
