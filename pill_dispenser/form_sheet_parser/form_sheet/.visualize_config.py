import json
import sys
import cv2

from form_sheet_parser import detect_aruco_corners, warp_form_sheet


def main(image_path: str, config_file: str = "form_sheet_config.json") -> None:
    img = cv2.imread(image_path)
    warped = warp_form_sheet(img, detect_aruco_corners(img))

    with open(config_file) as f:
        cfg = json.load(f)

    hw, hh = cfg["half_width"], cfg["half_height"]
    for group in cfg["groups"].values():
        for x, y in group["bubbles"].values():
            cv2.rectangle(warped, (x - hw, y - hh), (x + hw, y + hh),
                          (0, 255, 0), 1)

    cv2.imshow("verify", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(*sys.argv[1:])
