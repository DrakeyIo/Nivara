import os
import shutil
import sys
from pathlib import Path

from roboflow import Roboflow
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = Path(
    os.environ.get("YOLO_TRAIN_DIR", str(Path(sys.prefix).parent / "training"))
).expanduser().resolve()


def get_dataset_yaml() -> Path:
    local_data = os.environ.get("YOLO_DATA")
    if local_data:
        data_yaml = Path(local_data).expanduser().resolve()
        if not data_yaml.is_file():
            raise FileNotFoundError(f"YOLO_DATA does not point to a file: {data_yaml}")
        return data_yaml

    required = ("ROBOFLOW_API_KEY", "ROBOFLOW_WORKSPACE", "ROBOFLOW_PROJECT")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Set YOLO_DATA to a local data.yaml, or configure these environment "
            f"variables for Roboflow: {', '.join(missing)}"
        )

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    dataset = (
        Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
        .workspace(os.environ["ROBOFLOW_WORKSPACE"])
        .project(os.environ["ROBOFLOW_PROJECT"])
        .version(int(os.environ.get("ROBOFLOW_VERSION", "1")))
        .download("yolov8", location=str(TRAIN_DIR / "dataset"))
    )
    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Roboflow dataset has no data.yaml: {data_yaml}")
    return data_yaml


if __name__ == "__main__":
    dataset_yaml = get_dataset_yaml()
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(PROJECT_DIR / "yolo26n.pt"))
    model.train(
        data=str(dataset_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=2,
        project=str(TRAIN_DIR / "runs"),
        name="nivara",
    )
    shutil.copy2(model.trainer.best, PROJECT_DIR / "best.pt")
