"""Bounded, bidirectional instance-mask interpolation for captured frames."""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Frame:
    number: int
    time: float
    image: np.ndarray
    masks: list
    error: bool = False
    state: str = "DETECTED"
    anchors: str = ""
    captured_at: float = 0.0


def center(mask):
    y, x = np.nonzero(mask)
    return np.array([x.mean(), y.mean()])


def shift(mask, delta):
    return cv2.warpAffine(mask.astype(np.uint8),
                          np.float32([[1, 0, delta[0]], [0, 1, delta[1]]]),
                          (mask.shape[1], mask.shape[0]),
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0


def pair_masks(left, right, min_iou=0.15, max_distance=0.25):
    """Motion-aligned Mask IoU with explicit unmatched choices (Hungarian)."""
    n, m = len(left), len(right)
    cost = np.full((n + m, n + m), 1e6)
    cost[:n, m:] = 1
    cost[n:, :m] = 1
    cost[n:, m:] = 0
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            delta = center(b) - center(a)
            if np.linalg.norm(delta) > max_distance * np.hypot(*a.shape):
                continue
            moved = shift(a, delta)
            union = np.count_nonzero(moved | b)
            iou = np.count_nonzero(moved & b) / union if union else 0
            if iou >= min_iou:
                cost[i, j] = 1 - iou
    rows, cols = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(rows, cols)
            if i < n and j < m and cost[i, j] < 1e6]


class GapTracker:
    def __init__(self, max_gap=3, min_iou=0.15, max_distance=0.25):
        self.max_gap = max_gap
        self.min_iou = min_iou
        self.max_distance = max_distance
        self.left = None
        self.pending = []

    def push(self, item):
        if self.left and item.image.shape != self.left.image.shape:
            result = self.flush()
            self.left = None
            return result + self.push(item)
        # Errors are never treated as trustworthy empty detections.
        if item.error:
            result = self.flush()
            self.left = None
            item.state = "PROTECTED"
            return result + [item]
        if not item.masks:
            self.pending.append(item)
            if len(self.pending) > self.max_gap:
                result = self.flush()
                self.left = None
                return result
            return []
        result = []
        if self.pending:
            pairs = pair_masks(self.left.masks, item.masks, self.min_iou,
                               self.max_distance) if self.left else []
            # Partial matches cannot safely reconstruct all people.
            complete = (self.left is not None and len(pairs) == len(self.left.masks)
                        == len(item.masks) and item.time > self.left.time)
            for missing in self.pending:
                missing.state = "PROTECTED"
                if complete:
                    alpha = (missing.time - self.left.time) / (item.time - self.left.time)
                    if 0 < alpha < 1:
                        missing.masks = []
                        for i, j in pairs:
                            a, b = self.left.masks[i], item.masks[j]
                            delta = center(b) - center(a)
                            missing.masks.append(shift(a, alpha * delta) |
                                                 shift(b, -(1 - alpha) * delta))
                        missing.state = "TRACKED"
                        missing.anchors = f"{self.left.number},{item.number}"
                result.append(missing)
            self.pending = []
        self.left = item
        return result + [item]

    def flush(self):
        result, self.pending = self.pending, []
        for item in result:
            item.state = "PROTECTED"
        return result


def render(item, dilate=8):
    if item.state == "PROTECTED":
        return np.zeros_like(item.image)
    union = np.zeros(item.image.shape[:2], np.uint8)
    for mask in item.masks:
        union[mask > 0] = 255
    if dilate:
        k = 2 * dilate + 1
        union = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    out = item.image.copy()
    out[union > 0] = (40, 40, 40)
    return out
