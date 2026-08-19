"""Browse indexed slides in napari.

    uv run slideviz-view
    uv run slideviz-view --data /path/to/slides
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from slideviz.catalog import build, default_db
from slideviz.log import setup
from slideviz.settings import settings

log = logging.getLogger(__name__)


def main() -> None:
    """Index the data directory, then open napari with the slide list docked."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,  # keeps docstring layout
    )
    parser.add_argument("--data", type=Path, default=settings.data,  # --data still wins
                        help="directory of slides and their sidecars")
    # 0.01 s for 40 slides, so this is for when the set is much larger, or the data is away
    parser.add_argument("--no-reindex", action="store_true",
                        help="open the existing index instead of rebuilding it")
    parser.add_argument("-v", "--verbose", action="store_true", help="log at debug level")
    args = parser.parse_args()
    setup(args.verbose)

    if args.no_reindex and not default_db().exists():
        parser.error(f"no index at {default_db()}: run without --no-reindex first")

    if not args.no_reindex:
        if args.data is None:
            parser.error("no data directory: pass --data or set SLIDEVIZ_DATA")
        if not args.data.is_dir():
            parser.error(f"not a directory: {args.data}")

        count = build(args.data)  # rebuild first, so the list matches what is on disk
        log.info("indexed %d records from %s", count, args.data)

    import napari  # imported here, pulling in Qt is slow

    from slideviz.widget import SlideList

    viewer = napari.Viewer(title="slideviz")
    viewer.window.add_dock_widget(SlideList(viewer), name="Slides", area="right")
    viewer.scale_bar.visible = True
    napari.run()


if __name__ == "__main__":
    main()
