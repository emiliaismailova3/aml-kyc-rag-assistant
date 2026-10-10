"""Get plain text out of an invoice file (PDF or image).

Order of preference:
  1. A PDF with a text layer -> read the text directly with pdfplumber (fast, exact).
  2. Otherwise (scan, photo) -> preprocess the image and run Tesseract OCR.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pdfplumber
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# A PDF page with fewer characters than this is treated as "no text layer" (a scan).
MIN_TEXT_CHARS = 30
OCR_DPI = 300
OCR_LANGUAGES = "eng+aze"


def preprocess(image: Image.Image) -> Image.Image:
    """Grayscale -> deskew -> binarize. Makes Tesseract noticeably more accurate on scans."""
    gray = np.array(image.convert("L"))
    gray = _deskew(gray)
    # Otsu picks the black/white threshold automatically from the histogram.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Angle (degrees) by which the text is rotated, from the minimum-area box around the dark pixels."""
    inverted = cv2.bitwise_not(gray)
    _, mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 50:
        return 0.0
    angle = cv2.minAreaRect(points)[-1]
    # minAreaRect reports angles in different ranges depending on the OpenCV
    # version; fold into (-45, 45] so the result is the small tilt we want.
    if angle > 45:
        angle -= 90
    elif angle <= -45:
        angle += 90
    return float(angle)


def _deskew(gray: np.ndarray) -> np.ndarray:
    angle = estimate_skew_angle(gray)
    if abs(angle) < 0.3:
        return gray
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _configure_tesseract() -> None:
    """Find the Tesseract program: TESSERACT_CMD from .env/environment first, then the usual
    Windows install folder (the installer does not add it to PATH)."""
    import pytesseract

    command = os.getenv("TESSERACT_CMD", "").strip()
    default_windows = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
    if command:
        pytesseract.pytesseract.tesseract_cmd = command
    elif shutil.which("tesseract") is None and default_windows.exists():
        pytesseract.pytesseract.tesseract_cmd = str(default_windows)


def _ocr_languages() -> str:
    """Use eng+aze if the Azerbaijani model is installed, otherwise English only."""
    import pytesseract

    _configure_tesseract()
    installed = set(pytesseract.get_languages(config=""))
    wanted = [lang for lang in OCR_LANGUAGES.split("+") if lang in installed]
    if "aze" not in wanted:
        logger.warning("Tesseract language 'aze' is not installed; using %s only", wanted or ["eng"])
    return "+".join(wanted) or "eng"


def ocr_image(image: Image.Image) -> str:
    import pytesseract

    _configure_tesseract()
    return pytesseract.image_to_string(preprocess(image), lang=_ocr_languages())


def extract_text(path: str | Path) -> tuple[str, str]:
    """Return (text, method) where method is "text_layer" or "ocr"."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
            text = "\n".join(pages).strip()
            if len(text) >= MIN_TEXT_CHARS:
                return text, "text_layer"
            logger.info("%s has no text layer; falling back to OCR", path.name)
            images = [page.to_image(resolution=OCR_DPI).original for page in pdf.pages]
        return "\n".join(ocr_image(img) for img in images).strip(), "ocr"

    if suffix in IMAGE_SUFFIXES:
        with Image.open(path) as image:
            return ocr_image(image).strip(), "ocr"

    raise ValueError(f"Unsupported file type {suffix!r}; expected a PDF or an image")
