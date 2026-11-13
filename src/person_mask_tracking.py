"""Camera/video runner with bounded look-ahead for missing segmentation frames."""
import argparse
from collections import deque
import csv
from pathlib import Path
import time
import cv2
import numpy as np
from camera import open_source
from gap_tracking import Frame, GapTracker, render


def instance_masks(result, shape):
    if result.masks is None:
        if result.boxes is not None and len(result.boxes):
            raise ValueError("Person boxes without valid masks")
        return []
    # retina_masks=True gives masks in original image coordinates, including holes.
    data = result.masks.data.cpu().numpy()
    if data.shape[1:] != shape[:2] or not len(data) or any(not m.any() for m in data):
        raise ValueError("Invalid instance masks")
    return [m > 0.5 for m in data]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="0")
    p.add_argument("--weights", default=str(Path(__file__).resolve().parents[1] / "weights/yolov8n-seg.pt"))
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--max-gap", type=int, default=3)
    p.add_argument("--min-iou", type=float, default=0.15)
    p.add_argument("--max-distance", type=float, default=0.25, help="Maximum displacement / image diagonal")
    p.add_argument("--dilate", type=int, default=8)
    p.add_argument("--drop-frames", default="", help="Test only: discard masks at frame IDs, e.g. 2,4")
    p.add_argument("--save")
    p.add_argument("--log")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--fullscreen", action="store_true")
    args = p.parse_args()
    if args.max_gap < 1 or args.dilate < 0 or not 0 <= args.min_iou <= 1 or args.max_distance <= 0:
        p.error("Invalid tracking parameters")
    drops = {int(x) for x in args.drop_frames.split(",") if x.strip()}
    from ultralytics import YOLO
    model = YOLO(args.weights, task="segment")
    tracker = GapTracker(args.max_gap, args.min_iou, args.max_distance)
    cap = open_source(args.source, args.width, args.height, args.fps)
    if not cap.isOpened():
        cap.release()
        raise SystemExit("Cannot open source")
    is_file = Path(args.source).is_file()
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps or 15
    recent, missed = deque(maxlen=8), deque(maxlen=16)
    writer = log = None
    number = 0
    fields = ["frame", "state", "anchors", "inference_missed", "captured_s", "output_delay_ms"]
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log = open(args.log, "w", newline="", encoding="utf-8")
        csv_out = csv.DictWriter(log, fieldnames=fields)
        csv_out.writeheader()
    if args.fullscreen and not args.no_show:
        cv2.namedWindow("tracking", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def output(item):
        nonlocal writer
        recent.appendleft(f"#{item.number}:{item.state}")
        view = render(item, args.dilate)
        color = {"DETECTED": (0, 255, 0), "TRACKED": (0, 200, 255),
                 "PROTECTED": (0, 0, 255)}[item.state]
        panel = np.zeros((150, view.shape[1], 3), np.uint8)
        lines = [f"FRAME #{item.number}  {item.state}  anchors={item.anchors or '-'}",
                 "MISSED (newest first): " + ",".join(map(str, missed)),
                 "RECENT: " + " ".join(list(recent)[:4]),
                 " ".join(list(recent)[4:]),
                 "GREEN=detected YELLOW=tracked RED=protected"]
        for i, line in enumerate(lines):
            cv2.putText(panel, line, (8, 23 + 27 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
        view = np.vstack([panel, view])
        if args.save:
            if writer is None:
                Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (view.shape[1], view.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError("Cannot open video writer")
            writer.write(view)
        if log:
            csv_out.writerow(dict(frame=item.number, state=item.state, anchors=item.anchors,
                                  inference_missed=int(item.state != "DETECTED"),
                                  captured_s=item.time,
                                  output_delay_ms=round((time.monotonic() - item.captured_at) * 1000, 2)))
            log.flush()
        if not args.no_show:
            cv2.imshow("tracking", view)
            return cv2.waitKey(1) & 0xFF == ord("q")
        return False

    try:
        while True:
            ok, image = cap.read()
            if not ok and is_file:
                break
            number += 1
            captured_at = time.monotonic()
            timestamp = (number - 1) / fps if is_file else captured_at
            error, masks = not ok, []
            if ok:
                try:
                    result = model.predict(image, imgsz=args.imgsz, conf=args.conf,
                                           classes=[0], retina_masks=True, verbose=False)[0]
                    masks = instance_masks(result, image.shape)
                    if number in drops:
                        masks = []
                except Exception as exc:
                    print(f"#{number} inference error: {exc}")
                    error = True
            else:
                image = np.zeros((args.height, args.width, 3), np.uint8)
            item = Frame(number, timestamp, image, masks, error)
            item.captured_at = captured_at
            if not masks or error:
                missed.appendleft(number)
            stop = False
            for ready in tracker.push(item):
                stop = output(ready) or stop
            if stop or not ok:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for ready in tracker.flush():
            output(ready)
        cap.release()
        if writer:
            writer.release()
        if log:
            log.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
