"""
VisionLab - image_processing.py
Owner: Astik

All functions take a NumPy image array (BGR colour, as loaded by OpenCV,
or single-channel grayscale) and return a NEW NumPy array.
The input image is never modified.

Return formats:
    to_grayscale          -> 2D uint8 (H, W)
    gaussian_blur         -> same shape as input
    median_filter         -> same shape as input
    canny_edges           -> 2D uint8 (H, W), values 0 or 255
    otsu_threshold        -> 2D uint8 (H, W), values 0 or 255
    histogram_equalization-> same shape as input
"""

import cv2
import numpy as np


def _validate(image):
    """Check that the input is a usable image array."""
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy image array.")
    if image.size == 0:
        raise ValueError("Input image is empty.")
    if image.ndim not in (2, 3):
        raise ValueError("Image must be 2D (grayscale) or 3D (colour).")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError("Colour image must have 1, 3 or 4 channels.")


def _to_uint8(image):
    """Make sure the image is uint8 (0-255) so OpenCV functions work."""
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def _prepare(image):
    """Validate, copy, convert to uint8 and drop alpha channel if present."""
    _validate(image)
    img = _to_uint8(image).copy()
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    return img


def to_grayscale(image):
    """Convert a colour image to grayscale."""
    img = _prepare(image)
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(image, kernel_size=5):
    """Smooth the image with a Gaussian filter (reduces mild noise)."""
    img = _prepare(image)
    if kernel_size % 2 == 0:
        kernel_size += 1  # kernel size must be odd
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def median_filter(image, kernel_size=5):
    """Replace each pixel with the median of its neighbours
    (very good against salt-and-pepper / impulse noise)."""
    img = _prepare(image)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(img, kernel_size)


def canny_edges(image, low_threshold=100, high_threshold=200):
    """Detect edges using the Canny algorithm."""
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # reduce noise first
    return cv2.Canny(blurred, low_threshold, high_threshold)


def otsu_threshold(image):
    """Binary image using Otsu's automatic threshold selection."""
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def histogram_equalization(image):
    """Improve contrast. Grayscale: equalize directly.
    Colour: equalize only the brightness (Y) channel so colours stay natural."""
    img = _prepare(image)
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
