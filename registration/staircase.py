"""Remove scanner-written white background before processing."""

from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu
from valis.preprocessing import OD


def background_value(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove synthetic white background and flatten the remaining glass background."""
    grey = image.mean(axis=-1) if image.ndim == 3 else image
    white = grey >= 250  # synthetic, written where the grid was not scanned

    lit = grey[~white]
    if lit.size == 0:
        return np.ones_like(grey, bool), 255.0

    # tissue and glass are two well-separated modes, so Otsu finds the valley
    threshold = threshold_otsu(lit)
    glass = lit[lit >= threshold]
    return grey >= threshold, float(np.median(glass)) if glass.size else 255.0


def flatten_background(image: np.ndarray) -> np.ndarray:
    """Repaint every background pixel with one glass value, leaving tissue untouched."""
    background, glass = background_value(image)
    if not background.any():
        return image

    out = image.copy()
    out[background] = np.round(glass).astype(out.dtype)
    return out


class NoStaircase(OD):
    """VALIS's default brightfield processor, with the synthetic background removed first."""

    def process_image(self, *args, **kwargs):
        self.image = flatten_background(self.image)
        return super().process_image(*args, **kwargs)
