"""
Run:  python test_image_processing.py

Tests every function on:
  - every .jpg/.jpeg/.png in sample_images/ (put at least 3 there)
  - 3 synthetic images (so it works even with no sample images)
Saves outputs to test_outputs/ so you can look at them.
"""

import os
import glob
import cv2
import numpy as np
import image_processing as ip

FUNCS = {
    "grayscale": ip.to_grayscale,
    "gaussian": ip.gaussian_blur,
    "median": ip.median_filter,
    "canny": ip.canny_edges,
    "otsu": ip.otsu_threshold,
    "hist_eq": ip.histogram_equalization,
}

os.makedirs("test_outputs", exist_ok=True)


def synthetic_images():
    rng = np.random.default_rng(0)
    colour = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    cv2.rectangle(colour, (30, 30), (100, 90), (255, 255, 255), -1)
    gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
    low_contrast = (gray // 4 + 100).astype(np.uint8)
    return {"synthetic_colour": colour, "synthetic_gray": gray,
            "synthetic_lowcontrast": low_contrast}


def load_samples():
    imgs = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for path in glob.glob(os.path.join("sample_images", ext)):
            img = cv2.imread(path)
            if img is not None:
                imgs[os.path.splitext(os.path.basename(path))[0]] = img
    return imgs


def main():
    images = {**load_samples(), **synthetic_images()}
    failures = 0
    for img_name, img in images.items():
        original = img.copy()
        for fn_name, fn in FUNCS.items():
            try:
                out = fn(img)
                assert isinstance(out, np.ndarray), "output not ndarray"
                assert out.dtype == np.uint8, "output not uint8"
                assert out.shape[:2] == img.shape[:2], "size changed"
                assert np.array_equal(img, original), "INPUT WAS MODIFIED"
                cv2.imwrite(f"test_outputs/{img_name}_{fn_name}.png", out)
                print(f"PASS  {img_name:25s} {fn_name}")
            except Exception as e:
                failures += 1
                print(f"FAIL  {img_name:25s} {fn_name}: {e}")
    print("\nAll tests passed." if failures == 0 else f"\n{failures} failure(s).")


if __name__ == "__main__":
    main()
