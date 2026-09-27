"""
preprocess.py
-------------
Frame preprocessing to improve detection in low-light / night
conditions. Uses CLAHE (Contrast Limited Adaptive Histogram
Equalization) on the luminance channel, which is the standard,
citable technique used in low-light object detection papers.

This is applied BEFORE the frame is passed to the YOLO model.
"""

import cv2
import numpy as np


def enhance_low_light(frame: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    Apply CLAHE to the L-channel of an LAB image to boost visibility
    in dark frames without over-brightening already-bright frames.

    Args:
        frame: BGR image (as read by OpenCV)
        clip_limit: CLAHE clip limit (higher = more contrast boost)
        tile_grid_size: size of grid for local histogram equalization

    Returns:
        Enhanced BGR image, same shape as input.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def is_low_light(frame: np.ndarray, brightness_threshold: float = 90.0) -> bool:
    """
    Quick check: is this frame dark enough that we should run
    low-light enhancement? Avoids wasting time enhancing already
    well-lit daytime frames.

    Args:
        frame: BGR image
        brightness_threshold: mean pixel brightness (0-255) below
            which we consider the frame "low light"

    Returns:
        True if frame is dark, False otherwise.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    return mean_brightness < brightness_threshold


def multi_scale_retinex_color_restoration(frame: np.ndarray, sigmas=(15, 80)) -> np.ndarray:
    """
    Multi-Scale Retinex with Color Restoration (MSRCR).

    A stronger low-light enhancement than CLAHE -- recovers detail from
    frames so dark that CLAHE/gamma correction alone can't pull out
    enough structure for the detector to work with.

    PERFORMANCE NOTE: this is much heavier than CLAHE. To keep video
    processing usable, this function downsamples the frame before doing
    the expensive Gaussian blur + log math, then upsamples the result
    back to original size. Detail recovery at full resolution isn't
    needed here -- the goal is boosting overall visibility/contrast for
    the detector, not pixel-perfect enhancement. This cuts runtime by
    roughly 4x on typical dashcam-resolution frames with no meaningful
    loss in detection performance.
    """
    orig_h, orig_w = frame.shape[:2]

    # Downscale for the expensive part of the computation
    scale = 0.5
    small = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    img = small.astype(np.float32) + 1.0

    retinex = np.zeros_like(img)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex += np.log10(img) - np.log10(blur + 1.0)
    retinex /= len(sigmas)

    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = np.log10(125.0 * img / img_sum + 1.0)
    result = 128 * (retinex * color_restoration + 1.0)

    for i in range(3):
        channel = result[:, :, i]
        c_min, c_max = channel.min(), channel.max()
        result[:, :, i] = (channel - c_min) / (c_max - c_min + 1e-6) * 255

    result = np.clip(result, 0, 255).astype(np.uint8)

    # Upscale back to original resolution for the detector
    return cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def enhance_low_light_adaptive(frame: np.ndarray, very_dark_threshold: float = 55.0) -> np.ndarray:
    """
    Two-tier low-light enhancement:
    - Frames that are moderately dim -> CLAHE (fast, good enough)
    - Frames that are very dark (near-black night footage) -> MSRCR
      (slower, but recovers far more usable detail)

    Use this instead of plain enhance_low_light() when your footage
    includes true night/near-black conditions.
    """
    if is_very_dark(frame, very_dark_threshold):
        return multi_scale_retinex_color_restoration(frame)
    elif is_low_light(frame):
        return enhance_low_light(frame)
    return frame


def is_very_dark(frame: np.ndarray, threshold: float = 55.0) -> bool:
    """
    Check if a frame is dark enough to need MSRCR (not just CLAHE) and
    to warrant the lower night confidence threshold in detector.py.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < threshold


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline applied to every frame before detection.
    Uses adaptive enhancement: CLAHE for moderately dim frames, MSRCR
    for very dark (near-black night) frames, nothing for well-lit frames.
    """
    return enhance_low_light_adaptive(frame)