"""Browse indexed slides in napari.

    uv run slideviz-view
    uv run slideviz-view --data /path/to/slides
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slideviz.catalog import build
from slideviz.settings import settings


def main() -> None:
    """Index the data directory, then open napari with the slide list docked."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,  # keeps docstring layout
    )
    parser.add_argument("--data", type=Path, default=settings.data,  # --data still wins
                        help="directory of slides and their sidecars")
    args = parser.parse_args()

    if args.data is None:
        parser.error("no data directory: pass --data or set SLIDEVIZ_DATA")
    if not args.data.is_dir():
        parser.error(f"not a directory: {args.data}")

    count = build(args.data)  # rebuild first, so the list matches what is on disk
    print(f"indexed {count} slides from {args.data}")

    import napari  # imported here, pulling in Qt is slow

    from slideviz.widget import SlideList

    viewer = napari.Viewer(title="slideviz")
    viewer.window.add_dock_widget(SlideList(viewer), name="Slides", area="right")
    viewer.scale_bar.visible = True
    napari.run()


if __name__ == "__main__":
    main()
