from ultralytics import YOLO
from pathlib import Path

# === Пути ===
DATA_YAML = r"D:\CVLabs\Lab4\labeleddataset\YOLODataset\dataset.yaml"
PROJECT_DIR = r"D:\CVLabs\Lab4\runs\detect"

def train_yolo():
    model = YOLO("yolov8n.pt")  # можно заменить на yolov8s.pt

    model.train(
        data=DATA_YAML,
        epochs=30,
        imgsz=640,
        batch=16,
        project=PROJECT_DIR,
        name="car_detector_v1"
    )

    best_weights = Path(PROJECT_DIR) / "car_detector_v1" / "weights" / "best.pt"

    print(f"\n✅ Модель обучена!")
    print(f"📁 Весы: {best_weights}")

    return str(best_weights)

def predict_image(weights_path):
    test_image = r"D:\CVLabs\Lab4\test_car.png"  # изображение для теста

    model = YOLO(weights_path)

    model.predict(
        source=test_image,
        conf=0.25,
        save=True,
        project=PROJECT_DIR,
        name="car_prediction_v1"
    )

    print(f"\n📸 Результат сохранён в:")
    print(Path(PROJECT_DIR) / "car_prediction_v1")

if __name__ == "__main__":
    # 1. Обучение
    best_model = train_yolo()

    # 2. Предсказание
    predict_image(best_model)
