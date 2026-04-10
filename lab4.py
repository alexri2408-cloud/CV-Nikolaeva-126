import sys
import cv2
from tkinter import Tk, filedialog
from ultralytics import YOLO

# ===== Пути =====
MODEL_PATH = r"D:\CVLabs\Lab4\runs\detect\car_detector_v1\weights\best.pt"

def run_camera_detection(model, conf=0.25, camera_index=0):
    """
    Распознавание объектов с камеры в реальном времени.
    Нажми Q, чтобы выйти.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Не удалось открыть камеру.")
        return

    print("Камера запущена. Нажми 'Q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры.")
            break

        # Предсказание на текущем кадре
        results = model.predict(frame, conf=conf, verbose=False)

        # Рисуем боксы на кадре
        annotated_frame = results[0].plot()

        cv2.imshow("YOLO Camera Detection", annotated_frame)

        # Выход по Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def choose_image_file():
    """
    Открывает стандартное окно выбора файла Windows.
    Возвращает путь к выбранному изображению или None.
    """
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not file_path:
        return None

    return file_path

def run_image_detection(model, conf=0.25):
    """
    Выбор изображения через проводник Windows и распознавание на нём.
    """
    image_path = choose_image_file()

    if not image_path:
        print("Файл не выбран.")
        return

    print(f"Выбрано изображение: {image_path}")

    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print("Не удалось открыть изображение.")
        return

    # Предсказание
    results = model.predict(image, conf=conf, verbose=False)

    # Картинка с рамками
    annotated_image = results[0].plot()

    cv2.imshow("YOLO Image Detection", annotated_image)
    print("Нажми любую клавишу в окне изображения, чтобы закрыть его.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # При желании можно сохранить результат:
    save_path = image_path.rsplit(".", 1)[0] + "_detected.jpg"
    cv2.imwrite(save_path, annotated_image)
    print(f"Результат сохранён: {save_path}")

def main():
    model = YOLO(MODEL_PATH)

    while True:
        print("\nВыбери режим:")
        print("1 - Распознавание car с камеры")
        print("2 - Выбрать изображение и распознать car")
        print("0 - Выход")

        choice = input("Введите номер: ").strip()

        if choice == "1":
            run_camera_detection(model, conf=0.25, camera_index=0)
        elif choice == "2":
            run_image_detection(model, conf=0.25)
        elif choice == "0":
            print("Выход.")
            sys.exit()
        else:
            print("Неверный выбор. Попробуй ещё раз.")

if __name__ == "__main__":
    main()
