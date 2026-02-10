import os
import json
import pytest

# Import your function
from measured_full_body import cal_height


def _data_dir():
    return os.path.join(os.path.dirname(__file__), "data")


def _load_cases():
    path = os.path.join(_data_dir(), "golden_cases.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _has_required_files():
    """
    Skip tests if models are missing.
    Adjust paths if you keep models elsewhere.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    coco = os.path.join(project_root, "yolov8n.pt")
    spatula = os.path.join(project_root, "best.pt")
    return os.path.exists(coco) and os.path.exists(spatula)


@pytest.mark.skipif(not _has_required_files(), reason="Missing yolov8n.pt or best.pt in project root.")
@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["name"])
def test_infant_length_golden(case):
    img_path = os.path.join(_data_dir(), case["image"])
    assert os.path.exists(img_path), f"Missing test image: {img_path}"

    expected = float(case["expected_cm"])
    tol = float(case.get("tolerance_cm", 1.0))

    # Run your pipeline
    result = cal_height(img_path)
    print(img_path)
    print(result)
    assert result is not False and result is not None, "Measurement failed (returned False/None)."
    assert result > 0, f"Measurement returned non-positive value: {result}"

    # Compare within tolerance
    diff = abs(result - expected)
    assert diff <= tol, (
        f"{case['name']} failed: got {result:.2f} cm, expected {expected:.2f} ± {tol:.2f} "
        f"(diff={diff:.2f})"
    )
