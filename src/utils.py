"""
Shared utilities for the Manuscript Layout Region Detection pipeline.

Provides logging, path validation, image format handling, directory
management, and YAML configuration loading.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Any, Dict

import yaml

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


import contextlib
import sys


import io


@contextlib.contextmanager
def suppress_c_output(enable: bool = True):
    """
    Suppresses C-level and Python-level stdout/stderr messages
    (PaddleX, PIR passes, OneDNN, download notices, GLog) unless verbose=True.
    """
    if not enable:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()

    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        save_out = os.dup(1)
        save_err = os.dup(2)
    except Exception:
        yield
        return

    old_out = sys.stdout
    old_err = sys.stderr
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            os.dup2(save_out, 1)
            os.dup2(save_err, 2)
            os.close(null_fd)
            os.close(save_out)
            os.close(save_err)
        except Exception:
            pass
        sys.stdout = old_out
        sys.stderr = old_err


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure Python logging for the pipeline.
    Standard mode: clean, minimal terminal output.
    Verbose mode: detailed timestamped debug tracing across all pipeline stages.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s' if verbose
        else '%(message)s',
        datefmt='%H:%M:%S',
    )
    logger = logging.getLogger('manuscript-detector')
    logger.setLevel(log_level)

    if not logger.handlers:
        stream = getattr(sys.stdout, 'buffer', None)
        if stream is not None:
            out_stream = io.TextIOWrapper(stream, encoding='utf-8', errors='replace', line_buffering=True)
        else:
            out_stream = sys.stdout
        handler = logging.StreamHandler(out_stream)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for h in logger.handlers:
            h.setFormatter(formatter)
            h.setLevel(log_level)

    logger.propagate = False

    # Suppress third-party framework loggers unless in verbose mode
    if not verbose:
        for lib in ("ppocr", "paddlex", "paddle"):
            l = logging.getLogger(lib)
            l.setLevel(logging.ERROR)
            l.propagate = False

    return logger


def get_image_files(path: str) -> List[Path]:
    """
    Return a sorted list of valid image file paths from a file or directory.

    Args:
        path: Path to a single image file or a directory.

    Returns:
        List of Path objects pointing to valid image files.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If no valid images are found.
    """
    target = Path(path)

    if not target.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if target.is_file():
        if is_valid_image(str(target)):
            return [target]
        raise ValueError(
            f"Not a supported image format: {path}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if target.is_dir():
        files = sorted(
            f for f in target.iterdir()
            if f.is_file() and is_valid_image(str(f))
        )
        if not files:
            raise ValueError(f"No valid images in directory: {path}")
        return files

    raise ValueError(f"Path is neither file nor directory: {path}")


def is_valid_image(path: str) -> bool:
    """Check if a file has a supported image extension."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def ensure_dir(path: str) -> Path:
    """Create a directory (and parents) if it does not exist."""
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_device(requested_device: str = 'auto') -> str:
    """
    Resolve 'auto', 'cpu', or 'gpu' to an explicit device ('cpu' or 'gpu').

    - 'auto': returns 'gpu' when CUDA is compiled and available in Paddle, otherwise 'cpu'.
    - 'gpu': returns 'gpu' if CUDA is available, otherwise warns and safely falls back to 'cpu'.
    - 'cpu': returns 'cpu'.
    """
    req = (requested_device or 'auto').lower().strip()
    if req == 'cpu':
        return 'cpu'

    cuda_available = False
    try:
        import paddle
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            cuda_available = True
    except Exception:
        cuda_available = False

    if req == 'gpu':
        if cuda_available:
            return 'gpu'
        logging.getLogger('manuscript-detector').warning(
            "GPU requested via --device gpu, but CUDA is not available in Paddle. Falling back to CPU."
        )
        return 'cpu'

    # 'auto'
    return 'gpu' if cuda_available else 'cpu'


def configure_cpu_environment(cpu_threads: int = 6, opencv_threads: int = 2) -> None:
    """
    Configure thread limits across numerical and computer vision libraries
    to prevent thread contention, cache thrashing, and CPU oversubscription.
    """
    threads_str = str(max(1, cpu_threads))
    cv2_threads = max(1, opencv_threads)

    try:
        import cv2
        cv2.setNumThreads(cv2_threads)
    except Exception:
        pass

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = threads_str

    os.environ["GLOG_minloglevel"] = "3"
    os.environ["FLAGS_call_stack_level"] = "0"


def resolve_cpu_threads(configured_threads: Any = 'auto') -> int:
    """
    Resolve the operational CPU thread count:
    - If explicit int provided (e.g. 6) -> return that int clamped to logical cores.
    - If 'auto' or unspecified -> default to min(6, logical cores).
    """
    logical = os.cpu_count() or 4
    if isinstance(configured_threads, int) and configured_threads > 0:
        return min(configured_threads, logical)
    return min(6, logical)


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load YAML configuration from the default or specified path.

    Args:
        config_path: Optional explicit path.  Falls back to
                     config/default_config.yaml relative to the project root.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'config' / 'default_config.yaml'
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logging.getLogger('manuscript-detector').warning(
            f"Config not found: {config_path}. Using built-in defaults."
        )
        return _default_config()

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg else _default_config()


def load_class_mapping(mapping_path: Optional[str] = None) -> dict:
    """
    Load the class-mapping YAML that translates PP-DocLayoutV2 labels
    into the five assignment target classes.
    """
    if mapping_path is None:
        project_root = Path(__file__).parent.parent
        mapping_path = project_root / 'config' / 'class_mapping.yaml'
    else:
        mapping_path = Path(mapping_path)

    if not mapping_path.exists():
        logging.getLogger('manuscript-detector').warning(
            f"Class mapping not found: {mapping_path}. Using built-in defaults."
        )
        return _default_class_mapping()

    with open(mapping_path, 'r') as f:
        mapping = yaml.safe_load(f)
    return mapping if mapping else _default_class_mapping()


# Built-in fallbacks

def _default_config() -> dict:
    return {
        'model': {
            'name': 'PP-DocLayoutV2',
            'confidence_threshold': 0.15,
            'device': 'auto',
            'layout_nms': True,
        },
        'preprocessing': {
            'max_size': 1024,
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': 8,
            'deskew_max_angle': 15,
            'denoise': False,
        },
        'spatial_heuristics': {
            'header_zone': 0.18,
            'footer_zone': 0.82,
            'side_margin': 0.15,
        },
        'postprocessing': {
            'nms_iou_threshold': 0.45,
            'merge_iou_threshold': 0.65,
            'min_area_ratio': 0.003,
            'min_header_area_ratio': 0.0006,
            'min_footer_area_ratio': 0.001,
            'min_confidence': 0.10,
        },
        'visualization': {
            'colors': {
                'header': [50, 100, 255],
                'footer': [255, 200, 50],
                'main_text': [100, 255, 100],
                'side_text': [50, 200, 255],
                'filler': [255, 100, 200],
            },
            'line_thickness': 2,
            'line_thickness_ratio': 0.002,
            'font_scale_ratio': 0.001,
        },
        'confidence_scoring': {
            'weights': {
                'layout': 0.25,
                'ocr_det': 0.25,
                'ocr_rec': 0.20,
                'geometry': 0.15,
                'semantic': 0.15,
            },
        },
    }


def _default_class_mapping() -> dict:
    return {
        'direct_mappings': {
            'header': 'header',
            'footer': 'footer',
            'text': 'main_text',
        },
        'candidate_mappings': {
            'paragraph_title': 'header',
            'number': 'header',
            'aside_text': 'side_text',
            'reference': 'side_text',
            'reference_content': 'side_text',
            'footnote': 'side_text',
            'vision_footnote': 'side_text',
            'image': 'filler',
            'seal': 'filler',
            'chart': 'filler',
            'formula_number': 'filler',
            'inline_formula': 'filler',
            'table': 'main_text',
            'vertical_text': 'main_text',
        },
        'ignore_labels': [
            'abstract', 'algorithm', 'footer_image', 'header_image',
            'display_formula', 'figure_title', 'doc_title', 'content',
        ],
        'target_classes': ['header', 'main_text', 'footer', 'side_text', 'filler'],
    }
