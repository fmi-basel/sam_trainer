"""Remove 'sam_labels' entries from .zattrs files in zarr label folders."""

import json
from pathlib import Path

from tqdm import tqdm

ROOT = Path(r"./exp172-diff8.zarr")

for zattrs in tqdm(list(ROOT.rglob("labels/.zattrs"))):
    data = json.loads(zattrs.read_text())
    labels = data.get("labels", [])
    if "sam_labels" in labels:
        data["labels"] = [l for l in labels if l != "sam_labels"]
        zattrs.write_text(json.dumps(data, indent=2))
        print(f"Updated: {zattrs}")
