"""
Save what the wall model sees for one photo, so wall-finding problems can be
reproduced and fixed without the photo's owner re-running the app.

Usage (from the project folder, after the model has been downloaded once):
    python tools/wall_debug.py path\\to\\room.jpg

Writes next to the photo:
    <name>_walls.npz      model output (small) — send this file with the photo
    <name>_walls.jpg      raw model walls | cleaned-up walls | named walls
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.colour_studio import segmentation as seg  # noqa: E402
from src.colour_studio.recolour import outline_overlay  # noqa: E402


def main(photo: str) -> None:
    path = Path(photo)
    bgr = cv2.imread(str(path))
    if bgr is None:
        sys.exit(f"Can't read {path}")
    model = seg.find_local_model()
    if model is None:
        sys.exit("The wall model isn't downloaded yet — open Colour Studio in the app once first.")
    segm = seg.WallSegmenter.from_path(model)
    img = seg.to_working_size(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    probs = segm._probs(img)
    res = segm.segment(img)
    np.savez_compressed(
        path.with_name(path.stem + "_walls.npz"),
        probs=probs.astype(np.float16), image_shape=np.array(img.shape),
        id2label=np.array([f"{k}:{v}" for k, v in sorted(segm.id2label.items())]),
    )
    H, W = img.shape[:2]
    raw = cv2.resize(probs[segm.wall_ids].sum(axis=0).astype(np.float32), (W, H)) > 0.5
    teal = (11, 110, 121)
    panels = [
        outline_overlay(img, [(raw, teal, 0.35)], 2),
        outline_overlay(img, [(res.wall_mask, teal, 0.35)], 2),
        outline_overlay(img, [(r.mask, c, 0.35) for r, c in zip(res.regions, [(11, 110, 121), (217, 122, 43), (29, 78, 216), (180, 35, 24)] * 3)], 2),
    ]
    for p, label in zip(panels, ["model", "cleaned", ", ".join(r.name for r in res.regions)]):
        cv2.putText(p, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(p, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 1, cv2.LINE_AA)
    out = path.with_name(path.stem + "_walls.jpg")
    cv2.imwrite(str(out), cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR))
    print(f"Saved {out.name} and {path.stem}_walls.npz ({res.seconds:.1f}s, walls: {[r.name for r in res.regions]})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
