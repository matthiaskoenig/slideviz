"""Export every annotation polygon, with its source, from CVAT's database to JSON.

    python3 scripts/export_polygons.py <out.json>
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

CVAT = Path.home() / "Projects" / "cvat"

# runs inside cvat_server
SCRIPT = """
import json
from cvat.apps.engine.models import Image, LabeledShape, Task

task = Task.objects.get(name="necrosis overview")
images = {i.frame: i for i in Image.objects.filter(data=task.data)}

slides = {}
for shape in LabeledShape.objects.filter(
    job__segment__task=task, type="polygon"
).select_related("label"):
    image = images.get(shape.frame)
    if image is None:
        continue
    record = slides.setdefault(image.path, {
        "frame": shape.frame,
        "width": image.width,
        "height": image.height,
        "polygons": [],
    })
    record["polygons"].append({
        "label": shape.label.name,
        "source": shape.source,
        "points": list(shape.points),
    })

print(json.dumps(slides))
"""


def main() -> None:
    """Read the polygons out of CVAT and write them to one file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    result = subprocess.run(
        ["docker", "compose", "exec", "-T", "cvat_server",
         "python3", "/home/django/manage.py", "shell", "-c", SCRIPT],
        cwd=CVAT, capture_output=True, text=True, check=True,
    )
    slides = json.loads(result.stdout.strip().splitlines()[-1])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(slides, indent=2) + "\n")

    total = sum(len(s["polygons"]) for s in slides.values())
    print(f"{total} polygons across {len(slides)} slides -> {args.out}")
    for key in sorted(slides):
        record = slides[key]
        name = key.replace("mouse_apap_", "").replace("_he_s0.png", "")
        counts: dict[str, int] = {}
        for polygon in record["polygons"]:
            counts[polygon["source"]] = counts.get(polygon["source"], 0) + 1
        detail = ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
        print(f"  {name:<12} {len(record['polygons']):>4}  ({detail})")


if __name__ == "__main__":
    main()
