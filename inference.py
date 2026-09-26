#!/usr/bin/env python3
"""
Manuscript Layout Region Detection — CLI Entry Point

Uses PP-DocLayoutV2 (PaddleOCR) as the primary pretrained document-layout
model, combined with manuscript-aware heuristics and OpenCV refinement.

Usage:
    python inference.py --input ./data/test_images --output ./results
    python inference.py --input ./data/test_images/image.png --output ./results
    python inference.py --input ./data/test_images --output ./results --confidence 0.2 --verbose
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Any

# Suppress low-level C++ GLog and oneDNN diagnostic notices
os.environ["GLOG_minloglevel"] = "3"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

# Suppress benign third-party library warnings
warnings.filterwarnings('ignore')

from src.utils import (
    setup_logging, get_image_files, load_config,
    load_class_mapping, ensure_dir, suppress_c_output,
)
from src.preprocessor import preprocess, load_image, fix_orientation, restore_detections
from src.detector import ManuscriptDetector
from src.postprocessor import postprocess
from src.visualizer import draw_detections
from src.output_handler import save_annotated_image, save_json_metadata, print_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Manuscript Layout Region Detection (PP-DocLayoutV2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--input', required=True, help='Image file or directory.')
    p.add_argument('--output', default='./results', help='Output directory.')
    p.add_argument('--confidence', type=float, default=0.15,
                   help='Min confidence threshold (default: 0.15).')
    p.add_argument('--device', default='auto',
                   choices=['cpu', 'gpu', 'auto'])
    p.add_argument('--save-json', action='store_true', default=True)
    p.add_argument('--no-json', action='store_true', default=False)
    p.add_argument('--save-annotated', action='store_true', default=True)
    p.add_argument('--no-annotated', action='store_true', default=False)
    p.add_argument('--verbose', action='store_true', default=False)
    p.add_argument('--config', default=None, help='Custom YAML config.')
    return p.parse_args()


def process_image(image_path: str,
                   detector: ManuscriptDetector,
                   output_dir: str,
                   confidence: float,
                   config: dict,
                   save_json: bool,
                   save_ann: bool) -> dict:
    """Process one image through the full pipeline."""
    name = Path(image_path).name
    t0 = time.time()

    try:
        prep_cfg = (config or {}).get('preprocessing', {})
        if prep_cfg.get('enabled', False):
            processed, original, transform_info = preprocess(image_path, config)
            detections = detector.detect(image_path, processed, confidence)
            detections = restore_detections(detections, transform_info)
        else:
            original = load_image(image_path)
            original = fix_orientation(original, image_path)
            detections = detector.detect(image_path, original, confidence)

        oh, ow = original.shape[:2]
        tbs = getattr(detector, '_last_text_boxes', None)
        detections = postprocess(detections, ow, oh, confidence, config, image=original, text_boxes=tbs)

        elapsed = time.time() - t0

        if save_ann:
            ann = draw_detections(original, detections, config)
            save_annotated_image(ann, output_dir, name)

        if save_json:
            save_json_metadata(detections, name, (ow, oh), output_dir, elapsed)

        classes = {}
        for d in detections:
            classes[d['class']] = classes.get(d['class'], 0) + 1

        return {
            'image': name,
            'detections': len(detections),
            'time': elapsed,
            'classes': classes,
            'success': True,
        }

    except Exception as e:
        return {
            'image': name,
            'detections': 0,
            'time': time.time() - t0,
            'classes': {},
            'error': str(e),
            'success': False,
        }


def main():
    args = parse_args()
    logger = setup_logging(args.verbose)

    try:
        images = get_image_files(args.input)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    config = load_config(args.config)
    class_mapping = load_class_mapping()

    save_json = not args.no_json
    save_ann = not args.no_annotated

    ensure_dir(args.output)
    if save_ann:
        ensure_dir(os.path.join(args.output, 'annotated'))
    if save_json:
        ensure_dir(os.path.join(args.output, 'metadata'))

    with suppress_c_output(enable=not args.verbose):
        detector = ManuscriptDetector(config, class_mapping, device=args.device)

    if not detector.model_available:
        logger.warning("PP-DocLayoutV2 model is not available!")

    # Single startup summary
    sep = "-" * 60
    logger.info(sep)
    logger.info("Manuscript Layout Region Detection (PP-DocLayoutV2)")
    input_str = f"{args.input} ({len(images)} image{'s' if len(images) != 1 else ''})"
    logger.info(
        f"Model: PP-DocLayoutV2 | Device: {detector.device} | "
        f"Input: {input_str} | Output: {args.output} | Confidence: {args.confidence}"
    )
    logger.info(sep)

    total_dets, total_time = 0, 0.0
    results = []

    for i, img in enumerate(images, 1):
        with suppress_c_output(enable=not args.verbose):
            r = process_image(
                str(img), detector, args.output,
                args.confidence, config, save_json, save_ann,
            )
        results.append(r)
        total_dets += r['detections']
        total_time += r['time']

        dets_count = r['detections']
        dets_text = f"{dets_count} region" if dets_count == 1 else f"{dets_count} regions"
        if r['success']:
            logger.info(f"[{i}/{len(images)}] {img.name}  | {dets_text} | {r['time']:.1f}s")
        else:
            logger.error(f"[{i}/{len(images)}] {img.name}  | FAILED: {r.get('error', 'unknown error')}")

    ok = sum(1 for r in results if r['success'])
    print_summary(ok, total_dets, total_time, args.output)

    if ok < len(images):
        logger.warning(f"{len(images) - ok} image(s) failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
