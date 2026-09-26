"""Deterministic ownership regressions based on the supplied failure screenshots."""

import cv2
import numpy as np
import pytest

from src.semantic_refiner import SemanticRefiner


CASES = {
    "top-bottom-attribution": (1024, 1536, 0.22, 0.82),
    "top-strip-overreach": (1024, 1536, 0.22, 0.86),
    "compact-overlap": (238, 598, 0.12, 0.86),
    "bottom-body-overlap": (1024, 1536, 0.20, 0.87),
}


def component_line(y, x1, x2, text="", latin=0, deva=20):
    return {
        "bbox": {"x_min": x1, "y_min": y, "x_max": x2, "y_max": y + 16},
        "height": 16, "width": x2 - x1, "ocr_text": text, "ocr_conf": 0.96,
        "latin_count": latin, "devanagari_count": deva, "total_chars": latin + deva,
        "latin_ratio": latin / max(1, latin + deva),
        "deva_ratio": deva / max(1, latin + deva),
    }


@pytest.mark.parametrize("case_name", list(CASES.keys()))
def test_attached_failure_fixture_keeps_body_and_tight_attribution(case_name):
    """Use the real fixture dimensions with fixed, line-level ownership evidence."""
    h, w, body_start, body_end = CASES[case_name]
    image = np.full((h, w, 3), 240, dtype=np.uint8)
    body_ys = np.linspace(int(h * body_start), int(h * body_end), 6, dtype=int)
    for y in body_ys:
        cv2.putText(image, "Sanskrit text line in manuscript body", (int(w * 0.12), int(y + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    cv2.putText(image, "CC-0 In Public Domain Digitization by eGangotri", (int(w * 0.26), int(h * 0.03 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    cv2.putText(image, "CSU-Sringeri Campus Rare Book and Manuscript Collection", (int(w * 0.29), int(h * 0.94 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)

    refiner = SemanticRefiner()
    body = [component_line(int(y), int(w * 0.10), int(w * 0.90)) for y in body_ys]
    attribution = [
        component_line(
            int(h * 0.03), int(w * 0.25), int(w * 0.75),
            "CC-0 In Public Domain Digitization by eGangotri", latin=45, deva=0,
        ),
        component_line(
            int(h * 0.94), int(w * 0.28), int(w * 0.74),
            "CSU-Sringeri Campus Rare Book and Manuscript Collection", latin=52, deva=0,
        ),
    ]
    lines = refiner.attach_page_features(image, body + attribution, {"metrics": {"h_line": 16}}, [])
    owned = refiner.assign_ownership(lines, {"metrics": {"h_line": 16}}, image.shape[:2])
    regions = refiner.rebuild_owned_regions(image, owned)

    mains = [region for region in regions if region["class"] == "main_text"]
    fillers = [region for region in regions if region["class"] == "filler"]
    assert mains
    assert len(fillers) == 2
    assert max(region["bbox"]["y_max"] for region in mains) >= int(h * body_end)
    for filler in fillers:
        bbox = filler["bbox"]
        assert 0 <= bbox["x_min"] < bbox["x_max"] <= w
        assert 0 <= bbox["y_min"] < bbox["y_max"] <= h
        assert (bbox["x_max"] - bbox["x_min"]) < int(w * 0.65)
        assert (bbox["y_max"] - bbox["y_min"]) <= max(24, int(h * 0.08))
    for main in mains:
        for filler in fillers:
            assert main["bbox"]["y_max"] < filler["bbox"]["y_min"] or                    main["bbox"]["y_min"] > filler["bbox"]["y_max"]


def test_multiple_body_columns_and_footers_remain_independent():
    image = np.full((900, 1200, 3), 220, dtype=np.uint8)
    refiner = SemanticRefiner()
    left_body = [component_line(y, 220, 510) for y in (180, 215, 250, 285)]
    right_body = [component_line(y, 650, 980) for y in (180, 215, 250, 285)]
    footers = [component_line(790, 150, 260), component_line(790, 900, 1050)]
    lines = refiner.attach_page_features(
        image, left_body + right_body + footers, {"metrics": {"h_line": 16}}, []
    )
    owned = refiner.assign_ownership(lines, {"metrics": {"h_line": 16}}, image.shape[:2])
    regions = refiner.rebuild_owned_regions(image, owned)
    assert len([region for region in regions if region["class"] == "main_text"]) == 2
    assert len([region for region in regions if region["class"] == "footer"]) == 2


def deva_line(y, x1=180, x2=1180, text='धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः'):
    return {
        'bbox': {'x_min': x1, 'y_min': y, 'x_max': x2, 'y_max': y + 18},
        'height': 18, 'width': x2 - x1, 'ocr_text': text, 'ocr_conf': 0.92,
        'latin_count': 0, 'devanagari_count': 48, 'total_chars': 48,
        'latin_ratio': 0.0, 'deva_ratio': 1.0,
    }


def english_line(y, x1=450, x2=950, text='CSU-Sringeri Campus Rare Book and Manuscript Collection'):
    return {
        'bbox': {'x_min': x1, 'y_min': y, 'x_max': x2, 'y_max': y + 18},
        'height': 18, 'width': x2 - x1, 'ocr_text': text, 'ocr_conf': 0.94,
        'latin_count': 50, 'devanagari_count': 0, 'total_chars': 50,
        'latin_ratio': 1.0, 'deva_ratio': 0.0,
    }


def test_top_devanagari_title_is_header_not_filler():
    image = np.full((1000, 1400, 3), 235, dtype=np.uint8)
    lines = [
        deva_line(70, 520, 880, '॥ श्रीगणेशाय नमः ॥'),
        *[deva_line(y) for y in (170, 205, 240, 275, 310, 345)],
    ]
    refiner = SemanticRefiner()
    structure = {
        'content_frame': {'x_min': 100, 'y_min': 30, 'x_max': 1300, 'y_max': 950},
        'metrics': {'h_line': 18, 'g_vert': 12},
        'separators': {'horizontal': [], 'vertical': []},
    }
    lines = refiner.attach_page_features(image, lines, structure, [])
    owned = refiner.assign_ownership(lines, structure, image.shape[:2])
    regions = refiner.rebuild_owned_regions(image, owned)

    assert not [r for r in regions if r['class'] == 'filler']
    headers = [r for r in regions if r['class'] == 'header']
    mains = [r for r in regions if r['class'] == 'main_text']
    assert headers and mains
    assert headers[0]['bbox']['y_max'] < mains[0]['bbox']['y_min']


def test_no_footer_is_emitted_when_bottom_body_is_continuous():
    image = np.full((1000, 1400, 3), 235, dtype=np.uint8)
    lines = [deva_line(y) for y in (170, 205, 240, 275, 310, 345, 380, 415, 450, 485, 520, 555, 590, 625, 660, 695, 730, 765, 800)]
    refiner = SemanticRefiner()
    structure = {
        'content_frame': {'x_min': 100, 'y_min': 30, 'x_max': 1300, 'y_max': 950},
        'metrics': {'h_line': 18, 'g_vert': 17},
        'separators': {'horizontal': [], 'vertical': []},
    }
    lines = refiner.attach_page_features(image, lines, structure, [])
    owned = refiner.assign_ownership(lines, structure, image.shape[:2])
    regions = refiner.rebuild_owned_regions(image, owned)

    assert not [r for r in regions if r['class'] == 'footer']
    assert [r for r in regions if r['class'] == 'main_text']


def test_bottom_english_attribution_is_filler_not_footer():
    image = np.full((1000, 1400, 3), 235, dtype=np.uint8)
    lines = [deva_line(y) for y in (170, 205, 240, 275, 310, 345, 380, 415, 450, 485, 520, 555, 590, 625, 660, 695)]
    lines.append(english_line(900))
    cv2.putText(image, 'CSU-Sringeri Campus Rare Book and Manuscript Collection', (450, 916),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    refiner = SemanticRefiner()
    structure = {
        'content_frame': {'x_min': 100, 'y_min': 30, 'x_max': 1300, 'y_max': 950},
        'metrics': {'h_line': 18, 'g_vert': 17},
        'separators': {'horizontal': [], 'vertical': []},
    }
    lines = refiner.attach_page_features(image, lines, structure, [])
    owned = refiner.assign_ownership(lines, structure, image.shape[:2])
    regions = refiner.rebuild_owned_regions(image, owned)

    fillers = [r for r in regions if r['class'] == 'filler']
    assert fillers
    assert not [r for r in regions if r['class'] == 'footer']
    assert max(r['bbox']['y_max'] for r in fillers) > 890


def test_sanskrit_to_ascii_ocr_gibberish_is_not_filler():
    refiner = SemanticRefiner()
    gibberish = {
        'bbox': {'x_min': 200, 'y_min': 300, 'x_max': 1100, 'y_max': 318},
        'height': 18, 'width': 900,
        'ocr_text': 'MATEA.MNeTT 7QARVnN 0RFTLQX 9KTA',
        'ocr_conf': 0.58,
        'latin_count': 28, 'devanagari_count': 0, 'total_chars': 31,
        'latin_ratio': 28 / 31, 'deva_ratio': 0.0,
        'features': {'y_center_norm': 0.50, 'width_norm': 0.64},
    }
    assert refiner.calculate_filler_score(gibberish) == 0.0
    assert not refiner._has_positive_filler_evidence(gibberish)


def _line(x1, y1, x2, y2, flow_id, text='', deva=0.8, latin=0.0, conf=0.8):
    h = y2 - y1
    w = x2 - x1
    return {
        'bbox': {'x_min': x1, 'y_min': y1, 'x_max': x2, 'y_max': y2},
        'height': h,
        'width': w,
        'flow_id': flow_id,
        'ocr_text': text,
        'ocr_conf': conf,
        'deva_ratio': deva,
        'latin_ratio': latin,
        'latin_count': int(max(0, latin * 20)),
        'features': {'width_norm': w / 1536.0, 'y_center_norm': ((y1 + y2) / 2) / 1024.0},
    }


def test_header_prefix_is_split_from_body_flow():
    from src.semantic_refiner import SemanticRefiner
    refiner = SemanticRefiner(config={
        'role_assignment': {
            'header_prefix_max_lines': 3,
            'header_prefix_gap_ratio': 1.2,
            'macro_header_overlap_ratio': 0.25,
        },
        'semantic_refinement': {},
    })
    lines = [
        _line(560, 90, 850, 124, 'body', text='॥ श्रीगणेशाय नमः ॥', deva=0.7),
        _line(200, 160, 1250, 188, 'body', text='अथ योगानुशासनम्', deva=0.9),
        _line(200, 205, 1250, 233, 'body', text='योगश्चित्तवृत्तिनिरोधः', deva=0.9),
        _line(200, 250, 1250, 278, 'body', text='तदा द्रष्टुः स्वरूपेऽवस्थानम्', deva=0.9),
    ]
    structure = {
        'metrics': {'h_line': 28.0, 'g_vert': 20.0},
        'content_frame': {'x_min': 150, 'y_min': 60, 'x_max': 1300, 'y_max': 900},
        'separators': {'horizontal': [], 'vertical': []},
        'macro_proposals': [{'class': 'header', 'bbox': {'x_min': 540, 'y_min': 80, 'x_max': 870, 'y_max': 130}}],
    }
    owned = refiner.assign_ownership(lines, structure, (1024, 1536))
    assert owned[0]['owner'] == 'header'
    assert all(x['owner'] == 'main_text' for x in owned[1:])


def test_oversized_body_overlap_is_not_footer():
    from src.semantic_refiner import SemanticRefiner
    refiner = SemanticRefiner(config={'role_assignment': {}})
    lines = [
        _line(150, 180, 1400, 210, 'body', text='अथ...', deva=0.9),
        _line(150, 225, 1400, 255, 'body', text='योग...', deva=0.9),
        _line(150, 270, 1400, 300, 'body', text='चित्त...', deva=0.9),
        _line(150, 315, 1400, 345, 'body', text='वृत्ति...', deva=0.9),
        _line(140, 290, 1450, 880, 'footerish', text='सदृश ...', deva=0.8),
    ]
    structure = {
        'metrics': {'h_line': 30.0, 'g_vert': 15.0},
        'content_frame': {'x_min': 120, 'y_min': 120, 'x_max': 1460, 'y_max': 900},
        'separators': {'horizontal': [], 'vertical': []},
        'macro_proposals': [],
    }
    owned = refiner.assign_ownership(lines, structure, (1024, 1536))
    assert owned[-1]['owner'] != 'footer'


def test_oversized_macro_filler_is_removed_but_small_attribution_survives():
    from src.postprocessor import _semantic_hallucination_guard
    dets = [
        {'class': 'main_text', 'bbox': {'x_min': 150, 'y_min': 190, 'x_max': 1400, 'y_max': 870}, 'geometry_source': 'line_owned'},
        {'class': 'filler', 'bbox': {'x_min': 130, 'y_min': 170, 'x_max': 1450, 'y_max': 890}, 'geometry_source': 'macro_prior'},
        {'class': 'filler', 'bbox': {'x_min': 380, 'y_min': 110, 'x_max': 1150, 'y_max': 145}, 'geometry_source': 'macro_prior'},
        {'class': 'filler', 'bbox': {'x_min': 430, 'y_min': 900, 'x_max': 1110, 'y_max': 940}, 'geometry_source': 'macro_prior'},
    ]
    out = _semantic_hallucination_guard(dets, 1536, 1024)
    fillers = [d for d in out if d['class'] == 'filler']
    assert len(fillers) == 2
    assert not any(d['bbox']['y_min'] == 170 for d in fillers)


def test_non_line_owned_footer_inside_main_is_removed():
    from src.postprocessor import _semantic_hallucination_guard
    dets = [
        {'class': 'main_text', 'bbox': {'x_min': 130, 'y_min': 190, 'x_max': 1450, 'y_max': 890}, 'geometry_source': 'line_owned'},
        {'class': 'footer', 'bbox': {'x_min': 130, 'y_min': 774, 'x_max': 1450, 'y_max': 890}, 'geometry_source': 'macro_prior'},
        {'class': 'filler', 'bbox': {'x_min': 440, 'y_min': 910, 'x_max': 1100, 'y_max': 950}, 'geometry_source': 'line_owned'},
    ]
    out = _semantic_hallucination_guard(dets, 1536, 1024)
    assert not any(d['class'] == 'footer' for d in out)
    assert any(d['class'] == 'filler' for d in out)
