"""Display flattening: per-stain correction fields for slide-wide staining variation.

Fields are fitted on trusted tissue only and applied for display; measurements remain on raw pixels.
A field is a log factor per channel on a coarse grid: stain amounts are divided by exp(field) before
rebuilding the colour.

H&E: smooth 1 mm trend per channel, plus up to four soft-edged darker or paler regions.
DAB: smooth 1 mm counterstain trend; brown is modeled as a shared 2 mm trend with edge-rim removal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    binary_opening,
    distance_transform_edt,
    gaussian_filter,
    gaussian_filter1d,
    label,
    map_coordinates,
)
from scipy.optimize import minimize_scalar
from skimage.color import hed_from_rgb, rgb_from_hed
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import disk

VIEW_UM = 7.0  # resolution the fit reads; the nearest pyramid level is used
GRID_UM = 28.0  # field grid, about four view pixels
DAB_STAINS = ("cyp2e1", "cyp1a2")
CHANNELS = (0, 1, 2)  # skimage hed order: haematoxylin, eosin (residual on DAB slides), DAB

LUMEN_GREY = 215  # brighter view pixels inside the tissue are vessel openings
AMOUNT_FLOOR = 0.02  # stain amount below this is noise for the log
NECROSIS_MARGIN_UM = 100.0
EDGE_UM = 50.0  # outer band left out of the trend and region fits
OUTLINE_CLOSE_UM = 200.0  # depth is measured from the closed outer outline, not from holes
SUPPORT_UM, FULL_SUPPORT = 1000.0, 0.1  # full correction where >= 10 % of the surrounding mm is trusted
TREND_UM = 1000.0
CLIP = (0.25, 4.0)

REGION_TREND_UM = 3000.0
REGION_REACH_UM = 250.0  # support around trusted tissue where regions may lie
REGION_LEVELS_UM = (250.0, 120.0)  # seeding and boundary smoothing of the leftover map
REGION_HI, REGION_MIN_STEP, REGION_SHARP = 0.08, 0.06, 0.2  # log units
REGION_MIN_MM2, REGION_MAX_SHARE, MAX_REGIONS = 1.0, 0.5, 4
REGION_SEED_OPEN_UM, REGION_OPEN_UM, REGION_SMOOTH_UM, REGION_GROW_UM = 200.0, 150.0, 250.0, 500.0
REGION_RINGS_UM = ((50.0, 300.0), (300.0, 700.0))  # near and wide rings either side of a boundary
REGION_FEATHERS_UM = (100.0, 150.0, 250.0, 400.0)
REGION_FIT_BAND_UM = 600.0

BROWN_TREND_UM = 2000.0
RIM_SECTORS = 16
RIM_FIT_UM, RIM_BAND_UM = 1000.0, 50.0
RIM_TAPER_UM = (900.0, 1200.0)
DEEP_UM, DEEP_SHARE = 1200.0, 0.15  # a slide informs the shared decay when this much is that deep
RIM_QUANTILE = 25  # the rim scales weak and strong cells alike; weak cells ignore spared tissue
RIM_MAX = float(np.log(3.0))
RIM_FRACTION = 0.7  # removing all of the rim took too much brown off small sections
BROWN_CLIP = (0.4, 2.5)


@dataclass
class Grid:
    """What a fit needs from one slide, on the field grid."""

    logs: dict[int, np.ndarray]  # log stain amount per channel
    trust: np.ndarray  # living tissue away from lumina, necrosis and the edge band
    tissue: np.ndarray
    depth: np.ndarray  # um from the closed outer outline
    sector: np.ndarray  # sixteenth of the outline each pixel faces
    grid_um: float
    glass: np.ndarray


@dataclass
class Flattening:
    """Stored fields of one slide."""

    fields: dict[int, np.ndarray]
    tissue: np.ndarray
    grid_um: float
    glass: np.ndarray
    meta: dict = field(default_factory=dict)


def stain_amounts(rgb: np.ndarray, glass: np.ndarray) -> np.ndarray:
    """Haematoxylin, eosin and DAB amounts against the glass, unclipped."""
    od = -np.log(np.clip(np.asarray(rgb, np.float32) / glass, 1e-4, None))
    return od @ hed_from_rgb.astype(np.float32)


def to_rgb(amounts: np.ndarray, glass: np.ndarray) -> np.ndarray:
    """The colour image the stain amounts make on this glass."""
    return np.clip(glass * np.exp(-(amounts @ rgb_from_hed.astype(np.float32))), 0, 255).astype(np.uint8)


def pool(a: np.ndarray, k: int) -> np.ndarray:
    """Mean over k x k blocks, dropping the remainder."""
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:h, :w].reshape(h // k, k, w // k, k, *a.shape[2:]).mean(axis=(1, 3))


def masked_smooth(values: np.ndarray, weight: np.ndarray, sigma_px: float):
    """Normalised convolution: the weighted Gaussian mean, and the weight it rests on."""
    weight = np.asarray(weight, float)
    w = gaussian_filter(weight, sigma_px)
    return gaussian_filter(values * weight, sigma_px) / np.maximum(w, 1e-6), w


def trend(values: np.ndarray, trust: np.ndarray, sigma_px: float) -> np.ndarray:
    """Masked Gaussian, blended to three times wider where trusted tissue is thin."""
    near, w = masked_smooth(values, trust, sigma_px)
    far, _ = masked_smooth(values, trust, 3 * sigma_px)
    blend = np.clip(w / 0.2, 0, 1)
    return blend * near + (1 - blend) * far


def support(trust: np.ndarray, grid_um: float) -> np.ndarray:
    """0 to 1: trusted tissue in the surrounding mm, full from FULL_SUPPORT."""
    return np.clip(gaussian_filter(trust.astype(float), SUPPORT_UM / grid_um) / FULL_SUPPORT, 0, 1)


def prepare(view: np.ndarray, view_um: float, tissue: np.ndarray, necrotic: np.ndarray,
            glass: np.ndarray) -> Grid:
    """The fitting grid of one slide from its view at ~VIEW_UM, tissue and necrosis on that view."""
    k = max(1, round(GRID_UM / view_um))
    grid_um = view_um * k
    amounts = stain_amounts(view, glass)
    usable = tissue & (np.asarray(view, float).mean(axis=-1) < LUMEN_GREY)
    w = pool(usable.astype(float), k)
    tissue_g = pool(tissue.astype(float), k) > 0.5
    dead = binary_dilation(pool((necrotic & tissue).astype(float), k) > 0,
                           iterations=max(1, round(NECROSIS_MARGIN_UM / grid_um)))
    outline = binary_fill_holes(binary_closing(tissue_g, disk(max(1, round(OUTLINE_CLOSE_UM / grid_um)))))
    depth = distance_transform_edt(outline) * grid_um
    _, (rows, cols) = distance_transform_edt(outline, return_indices=True)  # nearest point outside
    centre = np.argwhere(outline).mean(axis=0)
    angle = np.arctan2(rows - centre[0], cols - centre[1])
    sector = np.floor((angle + np.pi) / (2 * np.pi) * RIM_SECTORS).astype(int) % RIM_SECTORS
    trust = tissue_g & ~dead & (w > 0.6) & (depth >= EDGE_UM)
    logs = {ch: pool(np.where(usable, np.log(np.clip(amounts[..., ch], AMOUNT_FLOOR, None)), 0), k)
            / np.maximum(w, 1e-6) for ch in CHANNELS}
    return Grid(logs, trust, tissue_g, depth, sector, grid_um, np.asarray(glass, float))


# regions: a smooth trend plus feathered offsets for coherent paler or darker regions


def _level(values: np.ndarray, trust: np.ndarray, sigma_px: float) -> np.ndarray:
    """Masked Gaussian, falling back to 3x and 10x wider scales and the median where trust is thin."""
    out = np.full(values.shape, float(np.median(values[trust])))
    for s in sorted({min(sigma_px * m, 150.0) for m in (10, 3, 1)}, reverse=True):
        v, w = masked_smooth(values, trust, s)
        b = np.clip(w / 0.2, 0, 1)
        out = b * v + (1 - b) * out
    return out


def _ring_median(d: np.ndarray, sel: np.ndarray) -> float:
    """Median of d over sel, NaN under 30 pixels."""
    return float(np.median(d[sel])) if sel.sum() >= 30 else np.nan


def _steps(d, trust, sup, mask, grid_um):
    """Near and wide step across the mask boundary, each with its midpoint."""
    inside = distance_transform_edt(mask | ~sup) * grid_um
    outside = distance_transform_edt(~mask) * grid_um
    out = []
    for lo, hi in REGION_RINGS_UM:
        a = _ring_median(d, trust & mask & (inside > lo) & (inside <= hi))
        b = _ring_median(d, trust & ~mask & sup & (outside > lo) & (outside <= hi))
        out += [a - b, (a + b) / 2]
    return out


def _fill_small_holes(mask: np.ndarray, min_px: float) -> np.ndarray:
    """The mask with holes under min_px filled."""
    holes = binary_fill_holes(mask) & ~mask
    labelled, n = label(holes)
    if n == 0:
        return mask
    sizes = np.bincount(labelled.ravel())
    return mask | (np.isin(labelled, np.nonzero(sizes < min_px)[0]) & (labelled > 0))


def _feathered(mask: np.ndarray, sup: np.ndarray, sigma_px: float) -> np.ndarray:
    """The mask carried past the support by the nearest support pixel, then blurred."""
    _, idx = distance_transform_edt(~sup, return_indices=True)
    return gaussian_filter(mask[idx[0], idx[1]].astype(float), sigma_px)


def _offsets(values, trust, feathers, grid_um, sel=None):
    """Step heights that best explain what the 1 mm trend leaves, Huber-weighted, and the loss."""
    sigma = TREND_UM / grid_um
    left = values - trend(values, trust, sigma)
    g = np.stack([f - trend(f, trust, sigma) for f in feathers], -1)
    if sel is None:
        sel = trust & (np.abs(g).max(-1) > 0.02)
    x, y = g[sel], left[sel]
    wt = np.ones(len(y))
    for _ in range(6):
        sw = np.sqrt(wt)
        coef, *_ = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)
        res = y - x @ coef
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
        wt = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(res), 1e-12))
    k, a = 1.345 * s, np.abs(res)
    return coef, float(np.mean(np.where(a <= k, 0.5 * a * a, k * (a - 0.5 * k))))


def _grow(d, r12, trust, sup, seed, sign, grid_um):
    """A region's boundary at the midpoint of its wide rings, kept near the seed and smoothed."""
    px = 1 / grid_um
    min_px = REGION_MIN_MM2 * 1e6 * px * px
    mid = _steps(d, trust, sup, seed, grid_um)[3]
    if not np.isfinite(mid):
        return None
    reach = binary_dilation(seed, disk(round(REGION_GROW_UM * px)))
    region, near = seed, np.nan
    for _ in range(2):
        region = (sign * (r12 - mid) > 0) & sup & reach
        region = (gaussian_filter(region.astype(float), REGION_SMOOTH_UM * px) > 0.5) & sup
        region = binary_opening(region, disk(round(REGION_OPEN_UM * px)))
        region = _fill_small_holes(region, min_px)
        labelled, _ = label(region)
        keep = np.unique(labelled[seed & region])
        region = np.isin(labelled, keep[keep > 0])
        if region.sum() < min_px:
            return None
        near, _, _, mid_new = _steps(d, trust, sup, region, grid_um)
        if np.isfinite(mid_new):
            mid = mid_new
    return region, near


def _best_region(d, values, trust, sup, grid_um):
    """The accepted region with the largest step, as (mask, step, feather um), or None."""
    px = 1 / grid_um
    min_px = REGION_MIN_MM2 * 1e6 * px * px
    r25 = _level(d, trust, REGION_LEVELS_UM[0] * px)
    r12 = _level(d, trust, REGION_LEVELS_UM[1] * px)
    best = None
    for sign in (1, -1):
        seeds = apply_hysteresis_threshold(np.where(sup, sign * r25, 0), REGION_HI / 2, REGION_HI)
        seeds = binary_opening(seeds, disk(round(REGION_SEED_OPEN_UM * px)))
        labelled, n = label(seeds)
        for i in range(1, n + 1):
            seed = labelled == i
            if seed.sum() < min_px:
                continue
            grown = _grow(d, r12, trust, sup, seed, sign, grid_um)
            if grown is None:
                continue
            region, near = grown
            if region.sum() > REGION_MAX_SHARE * sup.sum() or (trust & region).sum() < 0.3 * min_px:
                continue
            band = (trust & (distance_transform_edt(region | ~sup) * grid_um < REGION_FIT_BAND_UM)
                    & (distance_transform_edt(~region) * grid_um < REGION_FIT_BAND_UM))
            fits = []
            for feather in REGION_FEATHERS_UM:
                coef, loss = _offsets(values, trust, [_feathered(region, sup, feather * px)], grid_um, band)
                fits.append((loss, feather, float(coef[0])))
            _, feather, step = min(fits)
            if (np.isfinite(near) and sign * step >= REGION_MIN_STEP
                    and sign * near >= REGION_SHARP * sign * step
                    and (best is None or abs(step) > abs(best[1]))):
                best = (region, step, feather)
    return best


def region_field(log_c: np.ndarray, trust: np.ndarray, grid_um: float) -> np.ndarray:
    """Smooth 1 mm trend plus feathered step regions, chosen greedily, then refitted together."""
    px = 1 / grid_um
    sup = gaussian_filter(trust.astype(float), REGION_REACH_UM * px) > 0.02
    regions = []
    term = np.zeros_like(log_c)
    for _ in range(MAX_REGIONS):
        left = log_c - term
        best = _best_region(left - _level(left, trust, REGION_TREND_UM * px), left, trust, sup, grid_um)
        if best is None:
            break
        regions.append((best[0], best[2]))
        term = term + best[1] * _feathered(best[0], sup, best[2] * px)
    if regions:
        feathers = [_feathered(m, sup, f * px) for m, f in regions]
        coef, _ = _offsets(log_c, trust, feathers, grid_um)
        term = sum(c * f for c, f in zip(coef, feathers, strict=True))
    return trend(log_c - term, trust, TREND_UM * px) + term


# brown: an edge rim with a shared decay and a slow trend


def rim_shape(depth: np.ndarray, decay_um: float) -> np.ndarray:
    """exp(-depth / decay), tapered to zero between the RIM_TAPER_UM depths."""
    lo, hi = RIM_TAPER_UM
    return np.exp(-depth / decay_um) * np.clip((hi - depth) / (hi - lo), 0, 1)


def deep_mask(grid: Grid) -> np.ndarray:
    """Trusted tissue deeper than RIM_FIT_UM, or the deepest quarter of it on narrow sections."""
    deep = grid.trust & (grid.depth >= RIM_FIT_UM)
    if deep.sum() < 0.1 * grid.trust.sum():
        deep = grid.trust & (grid.depth >= np.percentile(grid.depth[grid.trust], 75))
    return deep


def deep_profile(grid: Grid) -> np.ndarray | None:
    """Median log DAB per depth band over the slide's deep median, or None when too little is deep."""
    deep = grid.trust & (grid.depth >= DEEP_UM)
    if deep.sum() < DEEP_SHARE * grid.trust.sum():
        return None
    logd = grid.logs[2]
    ref = np.median(logd[deep])
    return np.array([np.median(logd[grid.trust & (grid.depth >= e) & (grid.depth < e + RIM_BAND_UM)]) - ref
                     for e in np.arange(EDGE_UM, RIM_FIT_UM, RIM_BAND_UM)])


def rim_decay(profiles: list[np.ndarray]) -> float:
    """The shared decay length: exp(-depth / decay) fitted to the pooled normalised profiles."""
    norm = [p / np.mean(p[:2]) for p in profiles if np.mean(p[:2]) > 0.1]
    if not norm:
        raise ValueError("no slide has a measurable rim and enough deep tissue")
    pooled = np.median(np.array(norm), axis=0)
    depth = np.arange(EDGE_UM, RIM_FIT_UM, RIM_BAND_UM) + RIM_BAND_UM / 2
    first = depth[0]
    fit = minimize_scalar(lambda decay: np.sum((pooled - np.exp(-(depth - first) / decay)) ** 2),
                          bounds=(50.0, 2000.0), method="bounded")
    return float(fit.x)


def brown_parts(grid: Grid, decay_um: float, typical: float | None = None):
    """Slow trend (anchored on the deep tissue, scaled by support), rim, and rim strength per sector.

    Sectors without enough trusted tissue, or all of them when fewer than half have it, take
    `typical`, the median strength over the slides that inform the decay.
    """
    logd, trust, depth, sector = grid.logs[2], grid.trust, grid.depth, grid.sector
    g = rim_shape(depth, decay_um)
    deep = deep_mask(grid)
    bands = np.arange(EDGE_UM, RIM_FIT_UM, RIM_BAND_UM)
    rim = np.zeros_like(logd)
    strength = np.zeros(RIM_SECTORS)
    for _ in range(5):
        slow = trend(logd - rim, trust, BROWN_TREND_UM / grid.grid_um)
        left = logd - slow
        ref = np.percentile(left[deep], RIM_QUANTILE)
        strength = np.full(RIM_SECTORS, np.nan)
        for s in range(RIM_SECTORS):
            q, gb, n = [], [], []
            for e in bands:
                sel = trust & (sector == s) & (depth >= e) & (depth < e + RIM_BAND_UM)
                if sel.sum() >= 8:
                    q.append(np.percentile(left[sel], RIM_QUANTILE) - ref)
                    gb.append(np.mean(g[sel]))
                    n.append(sel.sum())
            if sum(n) >= 30:
                q, gb, n = np.array(q), np.array(gb), np.array(n)
                strength[s] = np.sum(n * q * gb) / np.sum(n * gb ** 2)
        if typical is not None and np.isfinite(strength).sum() < RIM_SECTORS // 2:
            strength[:] = np.nan
        fallback = typical if typical is not None else np.nanmedian(strength)
        strength = np.where(np.isfinite(strength), strength, fallback)
        strength = gaussian_filter1d(np.clip(strength, 0, RIM_MAX), 1.0, mode="wrap")
        rim = strength[sector] * g
    slow = (slow - np.median(slow[deep])) * support(trust, grid.grid_um)
    return slow, rim, strength


def _anchor(f: np.ndarray, trust: np.ndarray, clip: tuple[float, float]) -> np.ndarray:
    """The field shifted to zero at the slide's trusted median and clipped."""
    return np.clip(f - np.median(f[trust]), np.log(clip[0]), np.log(clip[1]))


def he_fields(grid: Grid) -> dict[int, np.ndarray]:
    """Regions fields for haematoxylin, eosin and the residual channel."""
    scale = support(grid.trust, grid.grid_um)
    return {ch: _anchor(region_field(grid.logs[ch], grid.trust, grid.grid_um), grid.trust, CLIP) * scale
            for ch in CHANNELS}


def dab_fields(grid: Grid, decay_um: float, typical: float | None) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Counterstain trend and the brown field (shared by the DAB and middle channels), with rim strengths."""
    counter = _anchor(trend(grid.logs[0], grid.trust, TREND_UM / grid.grid_um), grid.trust, CLIP)
    slow, rim, strength = brown_parts(grid, decay_um, typical)
    brown = _anchor(slow + RIM_FRACTION * rim, grid.trust, BROWN_CLIP)
    return {0: counter * support(grid.trust, grid.grid_um), 1: brown, 2: brown}, strength


def apply(rgb: np.ndarray, flat: Flattening, rgb_um: float, origin_um: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """An RGB patch with the fields divided out; origin_um is its top-left corner in slide um (y, x)."""
    rows, cols = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]].astype(float)
    coords = [((rows + 0.5) * rgb_um + origin_um[0]) / flat.grid_um - 0.5,
              ((cols + 0.5) * rgb_um + origin_um[1]) / flat.grid_um - 0.5]
    inside = map_coordinates(flat.tissue.astype(float), coords, order=1, mode="constant") > 0.5
    amounts = stain_amounts(rgb, flat.glass)
    for ch, f in flat.fields.items():
        amounts[..., ch] /= np.exp(map_coordinates(f, coords, order=1, mode="nearest"))
    # outside the tissue the raw pixels stay, so the background stays pure white for the viewer's mask
    return np.where(inside[..., None], to_rgb(amounts, flat.glass), rgb)


def flatten_levels(levels: list, flat: Flattening, level0_um: float) -> list:
    """A pyramid with the fields divided out, corrected chunk by chunk."""
    import dask.array as da

    def chunk(rgb: np.ndarray, block_info=None, level_um: float = 0.0) -> np.ndarray:
        """One chunk flattened, placed by its position in the level."""
        if not rgb.size:
            return rgb
        (row, _), (col, _) = block_info[0]["array-location"][:2]
        return apply(rgb, flat, level_um, (row * level_um, col * level_um))

    # stays lazy: only the chunks napari draws are corrected
    return [da.map_blocks(chunk, level, dtype=np.uint8,
                          level_um=level0_um * levels[0].shape[1] / level.shape[1])
            for level in levels]


def save(path: Path, flat: Flattening) -> None:
    """One slide's fields, tissue grid, glass and provenance as a compressed npz."""
    np.savez_compressed(path, **{f"field_{ch}": f.astype(np.float32) for ch, f in flat.fields.items()},
                        tissue=flat.tissue, grid_um=flat.grid_um, glass=flat.glass,
                        meta=json.dumps(flat.meta))


def load(path: Path) -> Flattening:
    """A slide's stored fields."""
    with np.load(path) as stored:
        fields = {int(name.removeprefix("field_")): stored[name].astype(float)
                  for name in stored.files if name.startswith("field_")}
        return Flattening(fields, stored["tissue"], float(stored["grid_um"]), stored["glass"],
                          json.loads(str(stored["meta"])))
