import json
import sys
import cv2
import numpy as np

from form_sheet_parser import (
    detect_aruco_corners,
    warp_form_sheet,
    SIDE,
)

GROUPS = [
    ("layer",    "one",  [str(i) for i in range(1, 9)]),
    ("days",     "many", ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]),
    ("duration", "one",  ["7d", "14d", "30d", "90d", "ongoing"]),
    ("hour",     "one",  [str(h) for h in range(24)]),
]

TASKS = [(g, m, k) for g, m, keys in GROUPS for k in keys]


def main(image_path: str, out_path: str = "form_sheet_config.json") -> None:
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"cannot read {image_path}")

    warped = warp_form_sheet(img, detect_aruco_corners(img))

    clicks: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(TASKS):
            clicks.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and clicks:
            clicks.pop()

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)

    while True:
        canvas = warped.copy()
        for (g, _, k), (x, y) in zip(TASKS, clicks):
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(canvas, f"{g}.{k}", (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        if len(clicks) < len(TASKS):
            g, _, k = TASKS[len(clicks)]
            label = f"click: {g}.{k}   ({len(clicks) + 1}/{len(TASKS)})"
        else:
            label = "done - press s to save, q to quit"

        cv2.rectangle(canvas, (0, 0), (SIDE, 26), (255, 255, 255), -1)
        cv2.putText(canvas, label, (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow("calibrate", canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and len(clicks) == len(TASKS):
            cfg = {
                "half_width": 8,
                "half_height": 16,
                "abs_min": 0.35,
                "margin": 0.15,
                "groups": {
                    g: {"mode": m, "bubbles": {}} for g, m, _ in GROUPS
                },
            }
            for (g, _, k), (x, y) in zip(TASKS, clicks):
                cfg["groups"][g]["bubbles"][k] = [x, y]

            with open(out_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"wrote {out_path}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(*sys.argv[1:])
