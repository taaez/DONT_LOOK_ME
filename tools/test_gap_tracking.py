"""Run: python -m unittest discover -s tools -p test_gap_tracking.py"""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gap_tracking import Frame, GapTracker, render, shift


def frame(number, x=None, error=False):
    image = np.full((60, 100, 3), 200, np.uint8)
    masks = []
    if x is not None:
        mask = np.zeros((60, 100), bool)
        mask[20:40, x:x + 10] = True
        masks = [mask]
    return Frame(number, float(number), image, masks, error)


class TrackingTests(unittest.TestCase):
    def test_one_three_five_reconstruct_two_four(self):
        tracker = GapTracker()
        outputs = []
        for item in [frame(1, 10), frame(2), frame(3, 20), frame(4), frame(5, 30)]:
            outputs.extend(tracker.push(item))
        self.assertEqual([x.number for x in outputs], [1, 2, 3, 4, 5])
        self.assertEqual([x.state for x in outputs],
                         ["DETECTED", "TRACKED", "DETECTED", "TRACKED", "DETECTED"])
        self.assertEqual(outputs[1].anchors, "1,3")
        self.assertTrue(outputs[1].masks[0][20:40, 15:25].all())
        self.assertTrue((render(outputs[1], 0)[20:40, 15:25] == 40).all())

    def test_timestamp_weighting(self):
        tracker = GapTracker()
        tracker.push(frame(1, 10))
        mid = frame(2)
        mid.time = 1.5
        tracker.push(mid)
        result = tracker.push(frame(3, 30))
        self.assertTrue(result[0].masks[0][20:40, 15:25].all())

    def test_unmatched_people_protect_entire_missing_frame(self):
        tracker = GapTracker()
        tracker.push(frame(1, 10))
        tracker.push(frame(2))
        right = frame(3, 20)
        right.masks.append(frame(3, 60).masks[0])
        missing = tracker.push(right)[0]
        self.assertEqual(missing.state, "PROTECTED")
        self.assertFalse(render(missing).any())

    def test_bounded_buffer_and_eof(self):
        tracker = GapTracker(max_gap=1)
        tracker.push(frame(1, 10))
        self.assertEqual(tracker.push(frame(2)), [])
        self.assertEqual([f.state for f in tracker.push(frame(3))],
                         ["PROTECTED", "PROTECTED"])
        tracker.push(frame(4))
        self.assertEqual(tracker.flush()[0].state, "PROTECTED")

    def test_error_breaks_anchor_chain(self):
        tracker = GapTracker()
        tracker.push(frame(1, 10))
        tracker.push(frame(2))
        self.assertEqual([f.state for f in tracker.push(frame(3, error=True))],
                         ["PROTECTED", "PROTECTED"])
        tracker.push(frame(4))
        self.assertEqual(tracker.push(frame(5, 20))[0].state, "PROTECTED")

    def test_translation_does_not_wrap(self):
        mask = frame(1, 90).masks[0]
        self.assertFalse(shift(mask, (20, 0)).any())


if __name__ == "__main__":
    unittest.main()
