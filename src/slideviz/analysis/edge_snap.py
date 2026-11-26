"""A map smoothed along an image's own edges: the colour guided filter (He, Sun and Tang, 2013).

Inside every window the output is a straight-line function of the guide's colours, fitted to the input
map, so the output keeps an edge where the guide has one and is smoothed where the guide is flat. Used to
move a coarse necrosis score onto the edges the H&E shows. Radius and smoothness are hand-set. It holds
about 300 bytes per pixel, so callers run a whole slide in blocks that overlap by twice the radius.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

# the necrosis map's guided filter: window radius in um and flat-window colour variance
NECROSIS_RADIUS_UM = 8.0
NECROSIS_EPS = 0.01


def window_mean(values: np.ndarray, radius: int) -> np.ndarray:
    """The mean over a (2 * radius + 1) square around every pixel, per channel, the border mirrored."""
    size = (2 * radius + 1, 2 * radius + 1, *([1] * (values.ndim - 2)))
    return uniform_filter(values, size=size, mode="reflect")


def guided_filter(guide: np.ndarray, source: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """The source map (H x W) filtered with an RGB guide (H x W x 3, values 0 to 1) over windows of 2 * radius + 1."""
    # eps is the colour variance below which a window counts as flat, where the map is simply averaged
    colour = guide.astype(np.float64)
    value = source.astype(np.float64)
    mean_colour = window_mean(colour, radius)
    mean_value = window_mean(value, radius)
    covariance = window_mean(colour * value[..., None], radius) - mean_colour * mean_value[..., None]

    # each window's 3 x 3 colour covariance, plus eps on the diagonal, which holds the slope near zero in flat windows
    products = colour[..., :, None] * colour[..., None, :]
    spread = window_mean(products.reshape(*colour.shape[:2], 9), radius).reshape(*colour.shape[:2], 3, 3)
    spread -= mean_colour[..., :, None] * mean_colour[..., None, :]
    spread += eps * np.eye(3)

    slope = np.linalg.solve(spread, covariance[..., None])[..., 0]
    offset = mean_value - (slope * mean_colour).sum(axis=-1)
    return (window_mean(slope, radius) * colour).sum(axis=-1) + window_mean(offset, radius)
