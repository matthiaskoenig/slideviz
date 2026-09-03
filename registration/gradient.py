"""Match on edges instead of colour.

Intensity-based feature matching fails across stains because the same tissue
looks different in H&E and DAB. Gradients preserve shared structure, which
improves alignment across stain pairs.
"""

from __future__ import annotations

import numpy as np
from skimage.filters import sobel
from staircase import flatten_background
from valis.preprocessing import OD


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Stretch to the full 0-255 range, which is what the detectors expect."""
    image = image.astype(float)
    spread = image.max() - image.min()
    if spread <= 0:
        return np.zeros(image.shape, np.uint8)
    return ((image - image.min()) / spread * 255).astype(np.uint8)


class GradientOD(OD):
    """Apply Sobel gradients after OD normalization; avoid smoothing."""

    def process_image(self, *args, **kwargs):
        self.image = flatten_background(self.image)
        processed = super().process_image(*args, **kwargs)
        return to_uint8(sobel(np.asarray(processed, float) / 255.0))


class SmoothGradientOD(GradientOD):
    """Blur before taking the gradient to suppress cell texture.

    This keeps shared tissue boundaries while reducing detector noise from
    stain-specific cellular patterns.
    """

    sigma = 2.0

    def process_image(self, *args, **kwargs):
        from skimage.filters import gaussian

        self.image = flatten_background(self.image)
        processed = OD.process_image(self, *args, **kwargs)
        smoothed = gaussian(np.asarray(processed, float) / 255.0, sigma=self.sigma)
        return to_uint8(sobel(smoothed))
