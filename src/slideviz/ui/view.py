"""Browse indexed slides in napari.

    uv run slideviz-view
    uv run slideviz-view --data /path/to/slides
    uv run slideviz-view --data /path/to/mouse /path/to/rat /path/to/species

Several data folders share one slide list; each slide's predictions, meshes and stain reference are found beside its
own folder.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from slideviz.data.catalog import build, default_db
from slideviz.log import setup
from slideviz.settings import settings

log = logging.getLogger(__name__)

# computed slide pieces kept for panning back; napari's default is a quarter of the RAM
DASK_CACHE_GB = 1.0
# threads in each calling thread's dask pool; the warp computes its source region
# inside a piece, which deadlocks a single shared pool
DASK_WORKERS = 4


def main() -> None:
    """Index the data directory, then open napari with the slide list docked."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,  # keeps docstring layout
    )
    parser.add_argument("--data", type=Path, nargs="+",  # SLIDEVIZ_DATA when --data is absent
                        default=[settings.data] if settings.data else None,
                        help="one or more directories of slides and their sidecars")
    # for a large set, or a slow or unreachable data directory
    parser.add_argument("--no-reindex", action="store_true",
                        help="skip the rebuild and open the existing index")
    parser.add_argument("--stain-reference", type=Path, default=None,
                        help="stain reference JSON; defaults to stain_reference.json beside each --data folder")
    parser.add_argument("-v", "--verbose", action="store_true", help="log at debug level")
    args = parser.parse_args()
    setup(args.verbose)
    console = Console(stderr=True)

    if args.data:
        settings.data = args.data[0]  # the default for tools that take one folder
    if args.stain_reference:
        settings.stain_reference = args.stain_reference

    if args.no_reindex and not default_db().exists():
        parser.error(f"no index at {default_db()}: run without --no-reindex first")

    if not args.no_reindex:
        if not args.data:
            parser.error("no data directory: pass --data or set SLIDEVIZ_DATA")
        for folder in args.data:
            if not folder.is_dir():
                parser.error(f"not a directory: {folder}")

        columns = (TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
                   TimeElapsedColumn())
        with Progress(*columns, console=console) as bar:
            for folder in args.data:
                # no total until the folder listing is done, so the bar pulses until then
                task = bar.add_task(f"indexing {folder}", total=None)
                count = build(folder, progress=lambda done, total, task=task: bar.update(
                    task, completed=done, total=total))  # rebuild first, so the list matches disk
                log.info("indexed %d records from %s", count, folder)

    # tiles are fetched off the UI thread, for this session only
    os.environ.setdefault("NAPARI_ASYNC", "1")
    with console.status("starting napari"):
        import dask
        import napari  # imported here, pulling in Qt is slow
        from napari.utils import resize_dask_cache

        from slideviz.ui.widget import SlideList

        dask.config.set(num_workers=DASK_WORKERS)
        viewer = napari.Viewer(title="slideviz")
        # after the viewer, which applies napari's own cache setting when it starts
        resize_dask_cache(int(DASK_CACHE_GB * 1e9))
        # --no-reindex lists the whole index
        widget = SlideList(viewer, None if args.no_reindex else args.data)
        viewer.window.add_dock_widget(widget, name="Slides", area="right")
        viewer.scale_bar.visible = True
    napari.run()


if __name__ == "__main__":
    main()
