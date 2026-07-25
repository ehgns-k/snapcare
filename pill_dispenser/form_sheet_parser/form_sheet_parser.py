import json
import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
SIDE = 1000

def generate_aruco_corners(out_path: str = "") -> None:
    """
    Generated aruco markers will be printed on a form sheet.
    IDs: 0=TL, 1=TR, 2=BR, 3=BL
    """
    for i in range(4):
        m = cv2.aruco.generateImageMarker(DICT, i, 600)
        m = cv2.copyMakeBorder(m, 120, 120, 120, 120,
                               cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(out_path + f"{i:02}_aruco.png", m)

def detect_aruco_corners(img: np.ndarray) -> np.ndarray:
    """
    Find aruco corners from form sheet image.
    Returns float32 (4, 2).
    Raises ValueError if less than 4 markers are found.
    """
    detector = cv2.aruco.ArucoDetector(DICT,
                                       cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(img)

    if ids is None:
        raise ValueError("no markers detected")

    found = {int(i): c[0] for i, c in zip(ids.flatten(), corners)}
    missing = [i for i in range(4) if i not in found]
    if missing:
        raise ValueError(f"missing marker ids: {missing}")

    return np.array([found[i][i] for i in range(4)], dtype=np.float32)

def warp_form_sheet(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """
    Warp the form sheet image to 1000x1000 image using detected corners.
    """
    dst = np.float32([[0, 0], [SIDE - 1, 0],
                      [SIDE - 1, SIDE - 1], [0, SIDE - 1]])
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (SIDE, SIDE))


def preprocess_form_sheet(warped: np.ndarray) -> np.ndarray:
    """
    Apply grayscale, Gaussian Blur, and adaptive thresholding
    to the warped image. output: ink = 255, paper = 0
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=51, C=10,
            )

def _fill_ratio(binary, cx, cy, hw, hh) -> float:
    roi = binary[cy - hh:cy + hh, cx - hw:cx + hw]
    return float(np.mean(roi)) / 255.0


def read_form_sheet(binary: np.ndarray,
                    config_file: str = "form_sheet_config.json") -> dict:
    """
    Read bubble groups using hardcoded coordinates.
    Returns {"values": {...}, "problems": [...]}.
    """
    with open(config_file) as f:
        cfg = json.load(f)

    hw, hh = cfg["half_width"], cfg["half_height"]
    abs_min, margin = cfg["abs_min"], cfg["margin"]

    values, problems = {}, []

    for name, group in cfg["groups"].items():
        scores = {
            key: _fill_ratio(binary, *xy, hw, hh)
            for key, xy in group["bubbles"].items()
        }

        if group["mode"] == "one":
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            best, second = ranked[0], ranked[1]
            if best[1] < abs_min:
                values[name] = None
                problems.append(f"{name}: nothing filled")
            elif best[1] - second[1] < margin:
                values[name] = None
                problems.append(
                    f"{name}: ambiguous ({best[0]}={best[1]:.2f}, "
                    f"{second[0]}={second[1]:.2f})"
                )
            else:
                values[name] = best[0]

        else:  # "many"
            picked = [k for k, v in scores.items() if v >= abs_min]
            unclear = [k for k, v in scores.items()
                       if abs_min - margin <= v < abs_min]
            values[name] = picked
            if not picked:
                problems.append(f"{name}: nothing filled")
            if unclear:
                problems.append(f"{name}: borderline {unclear}")

    return {"values": values, "problems": problems}
