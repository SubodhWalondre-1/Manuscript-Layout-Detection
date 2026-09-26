"""CLI regression tests that replace the heavyweight detector with a local fake."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import inference


class FakeDetector:
    """Small detector substitute so CLI tests never load or download models."""

    detect_calls = []

    def __init__(self, config, class_mapping, device='auto'):
        self.model_available = True
        self.device = 'cpu'
        self._last_text_boxes = None

    def detect(self, image_path, image, confidence):
        type(self).detect_calls.append(Path(image_path).name)
        h, w = image.shape[:2]
        return [{
            'class': 'main_text',
            'confidence': 0.91,
            'bbox': {'x_min': 1, 'y_min': 1, 'x_max': w - 1, 'y_max': h - 1},
        }]


class TestCli(unittest.TestCase):
    def _write_image(self, path: Path) -> None:
        image = np.full((32, 48, 3), 240, dtype=np.uint8)
        cv2.putText(image, 'x', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        self.assertTrue(cv2.imwrite(str(path), image))

    def _run_cli(self, arguments):
        FakeDetector.detect_calls = []
        with (
            patch.object(inference, 'ManuscriptDetector', FakeDetector),
            patch.object(inference, 'postprocess', side_effect=lambda dets, *args, **kwargs: dets),
            patch.object(sys, 'argv', ['inference.py', *arguments]),
        ):
            inference.main()

    def test_single_relative_image_writes_label_json_without_source_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            previous_cwd = os.getcwd()
            try:
                os.chdir(td)
                source = Path('source.png')
                self._write_image(source)
                before = hashlib.sha256(source.read_bytes()).hexdigest()

                self._run_cli(['--input', 'source.png', '--output', 'result'])

                self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())
                self.assertTrue(Path('result/annotated/source_annotated.png').is_file())
                metadata_path = Path('result/metadata/source_metadata.json')
                self.assertTrue(metadata_path.is_file())
                detection = json.loads(metadata_path.read_text(encoding='utf-8'))['detections'][0]
                self.assertIn('bbox', detection)
                self.assertIn('label', detection)
                self.assertIn('confidence', detection)
                self.assertNotIn('class', detection)
                self.assertEqual(detection['label'], 'main_text')
                self.assertIsInstance(detection['confidence'], float)
            finally:
                os.chdir(previous_cwd)

    def test_folder_input_processes_every_image(self):
        with tempfile.TemporaryDirectory() as td:
            previous_cwd = os.getcwd()
            try:
                os.chdir(td)
                source_dir = Path('input')
                source_dir.mkdir()
                self._write_image(source_dir / 'one.png')
                self._write_image(source_dir / 'two.jpg')

                self._run_cli(['--input', 'input', '--output', 'result', '--no-annotated'])

                self.assertEqual(FakeDetector.detect_calls, ['one.png', 'two.jpg'])
                self.assertTrue(Path('result/metadata/one_metadata.json').is_file())
                self.assertTrue(Path('result/metadata/two_metadata.json').is_file())
            finally:
                os.chdir(previous_cwd)

    def test_empty_and_invalid_inputs_exit_cleanly(self):
        with tempfile.TemporaryDirectory() as td:
            empty_dir = Path(td) / 'empty'
            empty_dir.mkdir()
            invalid_file = Path(td) / 'not-an-image.txt'
            invalid_file.write_text('not an image', encoding='utf-8')

            for input_path in (empty_dir, invalid_file):
                with self.subTest(input_path=input_path), patch.object(
                    sys, 'argv', ['inference.py', '--input', str(input_path)]
                ), self.assertRaises(SystemExit) as exited:
                    inference.main()
                self.assertEqual(exited.exception.code, 1)


    def test_arbitrary_relative_input_folder_and_output_folder(self):
        """1 & 3: Arbitrary relative input folder + arbitrary relative output folder, only annotated/ and metadata/."""
        with tempfile.TemporaryDirectory() as td:
            previous_cwd = os.getcwd()
            try:
                os.chdir(td)
                input_folder = Path('custom_in/sub_folder')
                input_folder.mkdir(parents=True)
                self._write_image(input_folder / 'ms_page_1.png')
                self._write_image(input_folder / 'ms_page_2.png')

                output_folder = Path('eval_output/eval_results')
                self._run_cli(['--input', str(input_folder), '--output', str(output_folder)])

                self.assertTrue(output_folder.is_dir())
                created_dirs = sorted(p.name for p in output_folder.iterdir())
                self.assertEqual(created_dirs, ['annotated', 'metadata'])

                self.assertTrue((output_folder / 'annotated' / 'ms_page_1_annotated.png').is_file())
                self.assertTrue((output_folder / 'annotated' / 'ms_page_2_annotated.png').is_file())
                self.assertTrue((output_folder / 'metadata' / 'ms_page_1_metadata.json').is_file())
                self.assertTrue((output_folder / 'metadata' / 'ms_page_2_metadata.json').is_file())
            finally:
                os.chdir(previous_cwd)

    def test_arbitrary_single_image_input_and_output_folder(self):
        """2: Arbitrary single-image input + arbitrary output folder."""
        with tempfile.TemporaryDirectory() as td:
            previous_cwd = os.getcwd()
            try:
                os.chdir(td)
                single_image = Path('arbitrary_dir/single_folio.png')
                single_image.parent.mkdir(parents=True)
                self._write_image(single_image)

                output_folder = Path('arbitrary_out')
                self._run_cli(['--input', str(single_image), '--output', str(output_folder)])

                self.assertTrue(output_folder.is_dir())
                created = sorted(p.name for p in output_folder.iterdir())
                self.assertEqual(created, ['annotated', 'metadata'])
                self.assertTrue((output_folder / 'annotated' / 'single_folio_annotated.png').is_file())
                self.assertTrue((output_folder / 'metadata' / 'single_folio_metadata.json').is_file())
            finally:
                os.chdir(previous_cwd)

    def test_no_debug_or_internal_folders_created(self):
        """4: No debug or internal folders are created anywhere in the output directory."""
        with tempfile.TemporaryDirectory() as td:
            previous_cwd = os.getcwd()
            try:
                os.chdir(td)
                input_dir = Path('data_in')
                input_dir.mkdir()
                self._write_image(input_dir / 'folio.png')

                output_dir = Path('run_out')
                self._run_cli(['--input', str(input_dir), '--output', str(output_dir)])

                forbidden_names = {
                    'adaptive_ownership_failures', 'internal_page_structure',
                    'cli_verification', 'debug', 'diagnostics', 'intermediate',
                    'temporary', 'logs'
                }
                all_subdirs = {p.name.lower() for p in output_dir.glob('**/*') if p.is_dir()}
                forbidden_found = all_subdirs & forbidden_names
                self.assertEqual(forbidden_found, set(), f"Forbidden folders created: {forbidden_found}")
            finally:
                os.chdir(previous_cwd)

    def test_input_remains_strictly_unchanged(self):
        """5: Original input images and directory remain completely unchanged."""
        with tempfile.TemporaryDirectory() as td:
            previous_cwd = os.getcwd()
            try:
                os.chdir(td)
                input_dir = Path('read_only_in')
                input_dir.mkdir()
                img1 = input_dir / 'test1.png'
                img2 = input_dir / 'test2.png'
                self._write_image(img1)
                self._write_image(img2)

                hash1_before = hashlib.sha256(img1.read_bytes()).hexdigest()
                hash2_before = hashlib.sha256(img2.read_bytes()).hexdigest()
                input_files_before = sorted(p.name for p in input_dir.iterdir())

                self._run_cli(['--input', str(input_dir), '--output', 'my_out'])

                hash1_after = hashlib.sha256(img1.read_bytes()).hexdigest()
                hash2_after = hashlib.sha256(img2.read_bytes()).hexdigest()
                input_files_after = sorted(p.name for p in input_dir.iterdir())

                self.assertEqual(hash1_before, hash1_after)
                self.assertEqual(hash2_before, hash2_after)
                self.assertEqual(input_files_before, input_files_after)
            finally:
                os.chdir(previous_cwd)

    def test_repeated_inference_does_not_create_extra_directories(self):
        """6: Running inference repeatedly into the same output folder creates no extra directories."""
        with tempfile.TemporaryDirectory() as td:
            previous_cwd = os.getcwd()
            try:
                os.chdir(td)
                input_dir = Path('input_repeat')
                input_dir.mkdir()
                self._write_image(input_dir / 'img.png')
                output_dir = Path('output_repeat')

                # First run
                self._run_cli(['--input', str(input_dir), '--output', str(output_dir)])
                first_run_items = sorted(p.name for p in output_dir.iterdir())
                self.assertEqual(first_run_items, ['annotated', 'metadata'])

                # Second run
                self._run_cli(['--input', str(input_dir), '--output', str(output_dir)])
                second_run_items = sorted(p.name for p in output_dir.iterdir())
                self.assertEqual(second_run_items, ['annotated', 'metadata'])
            finally:
                os.chdir(previous_cwd)

    def test_existing_unrelated_files_in_output_preserved(self):
        """Existing unrelated files in the output directory must not be deleted."""
        with tempfile.TemporaryDirectory() as td:
            previous_cwd = os.getcwd()
            try:
                os.chdir(td)
                input_dir = Path('input_dir')
                input_dir.mkdir()
                self._write_image(input_dir / 'page.png')

                output_dir = Path('existing_out')
                output_dir.mkdir()
                unrelated = output_dir / 'existing_note.txt'
                unrelated.write_text('important user note', encoding='utf-8')

                self._run_cli(['--input', str(input_dir), '--output', str(output_dir)])

                self.assertTrue(unrelated.is_file())
                self.assertEqual(unrelated.read_text(encoding='utf-8'), 'important user note')
                self.assertTrue((output_dir / 'annotated' / 'page_annotated.png').is_file())
                self.assertTrue((output_dir / 'metadata' / 'page_metadata.json').is_file())
            finally:
                os.chdir(previous_cwd)


if __name__ == '__main__':
    unittest.main()
