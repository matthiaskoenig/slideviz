"""Position in the liver lobule from the CYP2E1 signal, and stain profiles along it.

Averaged over tissue and blurred to lobule scale, the DAB signal is a landscape: peaks on the
pericentral zones, valleys on the periportal ones. A pixel's position is p = dp / (dp + dc), with dp
and dc its distances to the nearest valley and peak, or to the wall of a lumen beside one: 0
periportal, 1 pericentral. The formula is the portality of Albadry et al. 2024, whose portal
reference is the lobule boundary plus portal vessels; here it is the valleys, portal points as at
the corners of a hexagonal lobule. Distance to the watershed lines between the peaks' basins gives
the boundary reference for comparison. The basins are the lobules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.ndimage import label as label_regions
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from slideviz.analysis.dab import BAND_PX, LocalLevels

LANDSCAPE_UM = 7.0  # pixel size of the landscape grid; the nearest whole number of working pixels is used
BLOCK_TISSUE_MIN = 0.5  # share of a landscape pixel that must be tissue
SIGMA_UM = 50.0  # blur of the landscape: about one peak per pericentral zone
PEAK_GAP_UM = 60.0  # closest distance of two peaks, or of two valleys
EDGE_UM = 200.0  # profile pixels at least this far inside the section outline, whose edge stains brighter
LOBULE_BIN_MIN_UM2 = 10.0  # tissue a lobule needs in a bin for its own mean there
VESSEL_REACH_UM = 15.0  # a lumen this close to a peak or valley is its vessel, measured from its wall
VESSEL_MAX_UM2 = 30000.0  # larger glass-like areas inside the outline are tears or glass, never a vessel wall
WALL_UM = 12.0  # tissue this close to a lumen is vessel wall, its connective tissue and pixels half on glass, left out

POSITIONS = np.linspace(0, 1, 40)  # the positions of the SPT model's pattern scan
EDGES = np.concatenate([[0.0], (POSITIONS[:-1] + POSITIONS[1:]) / 2, [1.0]])  # bins centred on POSITIONS


def local_share(dab: np.ndarray, tissue: np.ndarray, levels: LocalLevels) -> np.ndarray:
    """Each pixel's DAB between its window's low (0) and high (1) level, uncut; 0 outside the tissue."""
    out = np.zeros(dab.shape, np.float32)
    cols = np.arange(dab.shape[1], dtype=np.float32)
    for r0 in range(0, dab.shape[0], BAND_PX):
        r1 = min(r0 + BAND_PX, dab.shape[0])
        rows, cc = np.meshgrid(np.arange(r0, r1, dtype=np.float32), cols, indexing="ij")
        low, high = levels.bounds(rows, cc)
        out[r0:r1] = np.where(tissue[r0:r1], (dab[r0:r1] - low) / np.maximum(high - low, 1e-6), 0)
    return out


def block_means(values: np.ndarray, weight: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean of values and mean weight over block x block squares; the last partial row and column dropped."""
    h, w = (values.shape[0] // block) * block, (values.shape[1] // block) * block
    shape = (h // block, block, w // block, block)
    wt = weight[:h, :w].astype(np.float32).reshape(shape).sum(axis=(1, 3))
    total = (values[:h, :w] * weight[:h, :w]).astype(np.float32).reshape(shape).sum(axis=(1, 3))
    return total / np.maximum(wt, 1e-6), wt / block**2


def on_working(a: np.ndarray, block: int, shape: tuple[int, int]) -> np.ndarray:
    """A landscape-grid array on the working grid, each landscape pixel over its block; the dropped rim takes the last one."""
    rows = np.minimum(np.arange(shape[0]) // block, a.shape[0] - 1)
    cols = np.minimum(np.arange(shape[1]) // block, a.shape[1] - 1)
    return a[np.ix_(rows, cols)]


@dataclass(frozen=True)
class Landscape:
    """The blurred DAB signal on the landscape grid, with its peaks and valleys."""

    field: np.ndarray  # blurred tissue-mean DAB per landscape pixel
    solid: np.ndarray  # landscape pixels that are mostly tissue
    usable: np.ndarray  # solid and at least EDGE_UM inside the outline, where profiles are read
    block: int  # working pixels per landscape pixel along each axis
    um_per_px: float  # landscape pixel size
    peaks: np.ndarray  # (n, 2) landscape rows and columns of the pericentral maxima
    valleys: np.ndarray  # (n, 2) landscape rows and columns of the periportal minima


def landscape(dab: np.ndarray, tissue: np.ndarray, inside: np.ndarray, um_per_px: float) -> Landscape:
    """Peaks and valleys of the tissue-averaged DAB blurred to lobule scale, from working-grid DAB, tissue and outline."""
    block = max(1, round(LANDSCAPE_UM / um_per_px))
    px = um_per_px * block
    mean, share = block_means(dab, tissue, block)
    solid = share >= BLOCK_TISSUE_MIN
    sigma = SIGMA_UM / px
    window = gaussian_filter(solid.astype(np.float32), sigma)
    field = gaussian_filter(np.where(solid, mean, 0).astype(np.float32), sigma) / np.maximum(window, 1e-3)
    within = block_means(inside, np.ones(inside.shape, bool), block)[0] >= 0.5
    # padded, so a section running off the image keeps its edge band at the frame
    usable = solid & (distance_transform_edt(np.pad(within, 1))[1:-1, 1:-1] * px >= EDGE_UM)

    # extrema of the whole solid field: every usable pixel keeps its nearest peak and valley, also in the edge band
    gap = max(1, round(PEAK_GAP_UM / px))
    labels = solid.astype(np.uint8)
    peaks = peak_local_max(field, min_distance=gap, labels=labels, exclude_border=False)
    valleys = peak_local_max(-field, min_distance=gap, labels=labels, exclude_border=False)
    if not len(peaks) or not len(valleys) or not usable.any():
        raise ValueError(f"{len(peaks)} peaks, {len(valleys)} valleys, {int(usable.sum())} usable landscape pixels")
    return Landscape(field, solid, usable, block, px, peaks, valleys)


@dataclass(frozen=True)
class Portality:
    """Position p per working pixel from points and from lobule borders, with each pixel's lobule."""

    points: np.ndarray  # float32, between the nearest valley (0) and peak (1)
    border: np.ndarray  # float32, between the nearest watershed line or portal lumen (0) and peak (1)
    lobule: np.ndarray  # int32, watershed basin of a peak, 0 where no basin reaches
    keep: np.ndarray  # working tissue pixels inside the usable landscape and off lumen walls, the pixels profiles use


def nearby_lumina(centres: np.ndarray, holes: np.ndarray, sizes: np.ndarray, reach: int, max_px: int) -> set[int]:
    """Labels of the glass-like areas within reach pixels of any centre, at most max_px pixels each."""
    found: set[int] = set()
    for r, c in centres:
        window = holes[max(0, r - reach): r + reach + 1, max(0, c - reach): c + reach + 1]
        found.update(int(k) for k in np.unique(window[window > 0]))
    return {k for k in found if sizes[k] <= max_px}


def portality(land: Landscape, tissue: np.ndarray, inside: np.ndarray) -> Portality:
    """Position of every working pixel between periportal (0) and pericentral (1), and its lobule."""
    shape = tissue.shape
    b = land.block
    um_per_px = land.um_per_px / b
    peak_centres = np.minimum(land.peaks * b + b // 2, np.array(shape) - 1)
    valley_centres = np.minimum(land.valleys * b + b // 2, np.array(shape) - 1)

    # lumina beside a peak are central veins, beside a valley portal vessels; one beside both is neither
    holes = label_regions(inside & ~tissue)[0]
    sizes = np.bincount(holes.ravel())
    reach = max(1, round(VESSEL_REACH_UM / um_per_px))
    max_px = round(VESSEL_MAX_UM2 / um_per_px**2)
    central = nearby_lumina(peak_centres, holes, sizes, reach, max_px)
    portal = nearby_lumina(valley_centres, holes, sizes, reach, max_px)
    central, portal = central - portal, portal - central

    def seeds(centres: np.ndarray, lumina: set[int]) -> np.ndarray:
        """Working pixels at the centres and in the given lumina."""
        mask = np.isin(holes, sorted(lumina)) if lumina else np.zeros(shape, bool)
        mask[tuple(centres.T)] = True
        return mask

    portal_seeds = seeds(valley_centres, portal)
    dc = distance_transform_edt(~seeds(peak_centres, central)).astype(np.float32)
    dp = distance_transform_edt(~portal_seeds).astype(np.float32)
    points = dp / np.maximum(dp + dc, 1e-6)
    del dp

    markers = np.zeros(land.field.shape, np.int32)
    markers[tuple(land.peaks.T)] = np.arange(1, len(land.peaks) + 1)
    lines = watershed(-land.field, markers, mask=land.solid, watershed_line=True)
    basins = watershed(-land.field, markers, mask=land.solid)
    # line pixels are those a basin reached; solid pieces without a peak stay unreached, outside every lobule
    line = on_working(land.solid & (lines == 0) & (basins > 0), b, shape)
    db = distance_transform_edt(~(line | portal_seeds)).astype(np.float32)
    border = db / np.maximum(db + dc, 1e-6)
    off_wall = distance_transform_edt(tissue | ~inside) * um_per_px > WALL_UM
    keep = tissue & off_wall & on_working(land.usable, b, shape)
    return Portality(points, border, on_working(basins, b, shape).astype(np.int32), keep)


@dataclass(frozen=True)
class Profile:
    """Sums of a stain and pixel counts per lobule (rows, row 0 outside every lobule) and position bin."""

    sums: np.ndarray
    counts: np.ndarray

    def mean(self) -> np.ndarray:
        """Mean stain per bin over all kept pixels."""
        return self.sums.sum(axis=0) / np.maximum(self.counts.sum(axis=0), 1)

    def share(self) -> np.ndarray:
        """Share of the kept tissue in each bin."""
        counts = self.counts.sum(axis=0)
        return counts / max(int(counts.sum()), 1)

    def lobule_means(self, min_count: int) -> np.ndarray:
        """Mean stain per lobule and bin, NaN where a lobule has fewer than min_count pixels in the bin."""
        with np.errstate(all="ignore"):
            return np.where(self.counts[1:] >= min_count, self.sums[1:] / self.counts[1:], np.nan)


def profile(p: np.ndarray, values: np.ndarray, lobule: np.ndarray, keep: np.ndarray) -> Profile:
    """Sums and counts of values per lobule and POSITIONS bin over the kept pixels."""
    bins = np.digitize(p[keep], EDGES[1:-1])
    n = int(lobule.max()) + 1
    index = lobule[keep].astype(np.int64) * len(POSITIONS) + bins
    size = n * len(POSITIONS)
    sums = np.bincount(index, values[keep].astype(np.float64), size).reshape(n, -1)
    counts = np.bincount(index, minlength=size).reshape(n, -1)
    return Profile(sums, counts)


def lobule_min_count(um_per_px: float) -> int:
    """Working pixels in LOBULE_BIN_MIN_UM2."""
    return max(1, round(LOBULE_BIN_MIN_UM2 / um_per_px**2))
