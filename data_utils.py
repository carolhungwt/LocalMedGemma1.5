"""
data_utils.py
─────────────
Handles all supported file types and converts them to a uniform list of
RGB PIL Images for downstream MedGemma inference.

Supported inputs
  • Standard images  : .jpg / .jpeg / .png / .bmp / .tiff
  • Video files      : .mp4 / .mov / .avi
  • DICOM slices     : .dcm  (single file OR a list of files forming a volume)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
import pydicom
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
SUPPORTED_DICOM_EXTENSIONS = {".dcm"}

DEFAULT_VIDEO_SLICES = 16   # uniformly spaced frames extracted from a video
MAX_DICOM_PREVIEW    = 64   # cap for preview slider performance


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Min-max normalise a float/int array to the 0-255 uint8 range."""
    arr = array.astype(np.float64)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        # Constant array — return zeros to avoid divide-by-zero
        return np.zeros_like(array, dtype=np.uint8)
    arr = (arr - lo) / (hi - lo) * 255.0
    return arr.astype(np.uint8)


def _apply_dicom_windowing(ds: pydicom.Dataset, pixel_array: np.ndarray) -> np.ndarray:
    """
    Apply RescaleSlope / RescaleIntercept if present, then normalise to uint8.
    Falls back to simple min-max normalisation when tags are absent.
    """
    arr = pixel_array.astype(np.float64)

    slope     = float(getattr(ds, "RescaleSlope",     1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    return _normalize_to_uint8(arr)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_image(file: BinaryIO | str | Path) -> list[Image.Image]:
    """
    Load a standard image file and return it as a single-element list of
    RGB PIL Images.

    Parameters
    ----------
    file : file-like object, str, or Path

    Returns
    -------
    list[PIL.Image.Image]  — always length 1
    """
    img = Image.open(file).convert("RGB")
    return [img]


def load_video(
    file: BinaryIO,
    num_slices: int = DEFAULT_VIDEO_SLICES,
) -> list[Image.Image]:
    """
    Extract *num_slices* uniformly spaced frames from a video file.

    Parameters
    ----------
    file       : file-like object (Streamlit UploadedFile or raw bytes buffer)
    num_slices : number of frames to sample

    Returns
    -------
    list[PIL.Image.Image]

    Raises
    ------
    ValueError  if the video cannot be opened or has no readable frames
    """
    # cv2 needs a filesystem path; write to a temporary bytes buffer instead
    raw_bytes = file.read() if hasattr(file, "read") else file
    np_buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    cap = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)  # single-frame test

    # For video we must use VideoCapture with a byte trick via imencode path
    # Write raw bytes to an in-memory file recognised by cv2 via a temp file
    import tempfile, os

    suffix = getattr(file, "name", ".mp4")
    suffix = Path(suffix).suffix if suffix else ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("cv2 could not open the video file.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError("Video appears to contain no readable frames.")

        # Clamp num_slices to total available frames
        num_slices = min(num_slices, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, num_slices, dtype=int)

        frames: list[Image.Image] = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            # BGR → RGB → PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

        cap.release()
    finally:
        os.unlink(tmp_path)

    if not frames:
        raise ValueError("No frames could be extracted from the video.")

    return frames


def load_dicom_file(file: BinaryIO) -> Image.Image:
    """
    Load a single DICOM file and return it as an RGB PIL Image.

    Parameters
    ----------
    file : file-like object

    Returns
    -------
    PIL.Image.Image  (RGB, uint8)
    """
    raw = file.read() if hasattr(file, "read") else file
    ds = pydicom.dcmread(io.BytesIO(raw))

    pixel_array = ds.pixel_array  # shape: (H, W) or (H, W, C)

    # Handle multi-frame DICOM (take the middle frame as representative)
    if pixel_array.ndim == 3 and pixel_array.shape[0] > 3:
        mid = pixel_array.shape[0] // 2
        pixel_array = pixel_array[mid]

    uint8_array = _apply_dicom_windowing(ds, pixel_array)

    # Grayscale → RGB
    if uint8_array.ndim == 2:
        pil_img = Image.fromarray(uint8_array, mode="L").convert("RGB")
    else:
        pil_img = Image.fromarray(uint8_array).convert("RGB")

    return pil_img


def load_dicom_series(files: list[BinaryIO]) -> list[Image.Image]:
    """
    Load an ordered list of DICOM files (a volume) and return each slice as
    an RGB PIL Image.

    The caller is responsible for sorting the files into anatomical order
    (e.g., by InstanceNumber or filename) before passing them here.

    Parameters
    ----------
    files : list of file-like objects (e.g., Streamlit UploadedFile list)

    Returns
    -------
    list[PIL.Image.Image]
    """
    images: list[Image.Image] = []
    for f in files:
        try:
            images.append(load_dicom_file(f))
        except Exception as exc:
            # Skip corrupt/non-image DICOM files (e.g., SR, PR objects)
            import warnings
            warnings.warn(f"Skipping DICOM file due to error: {exc}")
    return images


def process_uploaded_files(
    uploaded_files: list,
    num_video_slices: int = DEFAULT_VIDEO_SLICES,
) -> tuple[list[Image.Image], str]:
    """
    Top-level dispatcher: inspect the uploaded file(s) and return the
    appropriate list of PIL Images alongside a human-readable modality label.

    Parameters
    ----------
    uploaded_files    : list of Streamlit UploadedFile objects
    num_video_slices  : frames to extract for video inputs

    Returns
    -------
    (images, modality_label)
      images         : list[PIL.Image.Image]
      modality_label : str  — e.g. "Image", "Video (16 frames)", "DICOM (32 slices)"

    Raises
    ------
    ValueError  for unsupported file types or empty results
    """
    if not uploaded_files:
        raise ValueError("No files were provided.")

    # Normalise to list
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    first_name = Path(uploaded_files[0].name).suffix.lower()

    # ── DICOM series ────────────────────────────────────────────────────────
    if first_name in SUPPORTED_DICOM_EXTENSIONS:
        # Sort by filename; a more robust approach would sort by InstanceNumber
        sorted_files = sorted(uploaded_files, key=lambda f: f.name)
        images = load_dicom_series(sorted_files)
        if not images:
            raise ValueError("No valid DICOM slices could be decoded.")
        return images, f"DICOM ({len(images)} slice{'s' if len(images) != 1 else ''})"

    # ── Single-file inputs ───────────────────────────────────────────────────
    single_file = uploaded_files[0]
    ext = Path(single_file.name).suffix.lower()

    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        images = load_image(single_file)
        return images, "Image"

    if ext in SUPPORTED_VIDEO_EXTENSIONS:
        images = load_video(single_file, num_slices=num_video_slices)
        return images, f"Video ({len(images)} frames)"

    raise ValueError(
        f"Unsupported file type: '{ext}'. "
        f"Accepted formats: images {sorted(SUPPORTED_IMAGE_EXTENSIONS)}, "
        f"video {sorted(SUPPORTED_VIDEO_EXTENSIONS)}, "
        f"DICOM {sorted(SUPPORTED_DICOM_EXTENSIONS)}."
    )
