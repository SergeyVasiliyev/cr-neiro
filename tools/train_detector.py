"""Trains the unit detector (YOLO, ultralytics) on raw/detector/dataset.

    .venv\\Scripts\\python.exe tools\\train_detector.py [--model yolo11s.pt] [--epochs 150] [--imgsz 960]

Starts from COCO-pretrained weights. Horizontal flips are fine (the arena is left-right symmetric),
vertical flips are not (they would swap our side and theirs). Results: raw/detector/runs/<name>/,
best weights in weights/best.pt.
"""
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join(ROOT, "raw", "detector", "dataset", "data.yaml"))
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--name", default="units")
    p.add_argument("--patience", type=int, default=60, help="stop after this many epochs without improvement")
    p.add_argument("--resume", action="store_true", help="continue the stopped run --name from its last.pt")
    args = p.parse_args()

    from ultralytics import YOLO

    if args.resume:
        YOLO(os.path.join(ROOT, "raw", "detector", "runs", args.name, "weights", "last.pt")).train(resume=True)
        return

    weights_dir = os.path.join(ROOT, "raw", "detector", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    model = YOLO(os.path.join(weights_dir, args.model) if os.path.exists(os.path.join(weights_dir, args.model))
                 else args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=0, workers=2,
        project=os.path.join(ROOT, "raw", "detector", "runs"), name=args.name, exist_ok=True,
        fliplr=0.5, flipud=0.0, degrees=0.0, mosaic=1.0, close_mosaic=20, patience=args.patience, plots=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
