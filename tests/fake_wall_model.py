"""
A stand-in for the SegFormer ONNX session, for tests.

It knows the true label map of a sample room and returns logits that make the
same kinds of mistakes the real model makes on real photos:
- coarse: one prediction per 4x4 input pixels, so edges are blocky
- bites:  bright, lamp-lit wall near the ceiling labelled "ceiling"
- holes:  a few blobs inside the wall labelled "ceiling"
- leaks:  the wall label spilling a little way onto the headboard / sofa back
- ragged: blotchy "curtain" patches on one side wall
"""
import cv2
import numpy as np

WALL, CEILING, CURTAIN = 0, 5, 18
FURNITURE_TOPS = (7, 23)  # bed, sofa


class _Inp:
    name = "pixel_values"


class FakeSegFormer:
    def __init__(self, image_rgb: np.ndarray, labels: np.ndarray, mistakes: bool = True, seed: int = 0,
                 square_only: bool = False, n_classes: int = 150):
        self.image = image_rgb
        self.truth = labels.astype(np.int32)
        self.n_classes = n_classes
        self.square_only = square_only
        self.calls = []
        self.pred = self.truth.copy()
        if mistakes:
            self._add_mistakes(np.random.default_rng(seed))
        # Where the model is unsure (bites/holes), wall still gets some probability
        self.unsure = (self.pred != self.truth) & (self.truth == WALL)

    def _add_mistakes(self, rng):
        h, w = self.truth.shape
        wall = self.truth == WALL
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        top_band = np.zeros_like(wall)
        top_band[: int(0.45 * h)] = True
        # Bites: bright wall near the ceiling
        bright = gray > np.percentile(gray[wall], 90)
        bites = wall & top_band & cv2.dilate(bright.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        self.pred[bites] = CEILING
        # Holes: a few blobs in the middle of the wall
        for _ in range(4):
            ys, xs = np.nonzero(wall & ~bites)
            i = rng.integers(len(xs))
            r = int(rng.integers(10, 22))
            blob = np.zeros((h, w), np.uint8)
            cv2.ellipse(blob, (int(xs[i]), int(ys[i])), (r, int(r * 0.7)), float(rng.integers(0, 180)), 0, 360, 1, -1)
            self.pred[(blob > 0) & wall] = CEILING
        # Leaks: wall label spills 4% of the height onto the top of furniture
        for cls in FURNITURE_TOPS:
            obj = self.truth == cls
            if not obj.any():
                continue
            top = obj.argmax(axis=0)
            has = obj.any(axis=0)
            band = int(0.04 * h)
            for x in np.nonzero(has)[0]:
                wobble = int(band * (0.5 + 0.5 * np.sin(x / 9.0)))
                self.pred[top[x]: top[x] + wobble, x] = WALL
        # Ragged: blotchy curtain patches on the right-hand side wall
        noise = cv2.GaussianBlur(rng.normal(0, 1, (h // 16, w // 16)).astype(np.float32), (0, 0), 1.2)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        right = wall & (np.arange(w)[None, :] > 0.82 * w)
        self.pred[right & (noise > 0.35)] = CURTAIN

    def get_inputs(self):
        return [_Inp()]

    def _is_flipped(self, x: np.ndarray) -> bool:
        img = np.transpose(x[0], (1, 2, 0))
        g = img.mean(axis=2)
        ref = cv2.resize(cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY).astype(np.float32), (g.shape[1], g.shape[0]))
        a = np.corrcoef(g.ravel(), ref.ravel())[0, 1]
        b = np.corrcoef(g.ravel(), ref[:, ::-1].ravel())[0, 1]
        return b > a

    def run(self, _outputs, feeds):
        x = feeds["pixel_values"]
        _, _, ih, iw = x.shape
        self.calls.append((ih, iw))
        if self.square_only and ih != iw:
            raise RuntimeError("fake model: only square inputs supported")
        oh, ow = ih // 4, iw // 4
        pred = cv2.resize(self.pred.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        unsure = cv2.resize(self.unsure.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST).astype(bool)
        if self._is_flipped(x):
            pred, unsure = pred[:, ::-1], unsure[:, ::-1]
        logits = np.zeros((1, self.n_classes, oh, ow), np.float32)
        np.put_along_axis(logits[0], pred[None], 6.0, axis=0)
        logits[0, WALL][unsure] = 4.5          # the model wavers on bites and holes
        logits += np.random.default_rng(1).normal(0, 0.3, logits.shape).astype(np.float32)
        return [logits]


LABELS = {0: "wall", 3: "floor", 5: "ceiling", 7: "bed", 8: "windowpane", 14: "door", 15: "table", 18: "curtain",
          19: "chair", 22: "painting", 23: "sofa", 28: "rug", 36: "lamp", 39: "cushion", 57: "pillow"}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    return float((a & b).sum() / max((a | b).sum(), 1))
