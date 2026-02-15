import cv2
import argparse

# Глобальные переменные
rectangles = []
RECT_SIZE = 40

# Обработчик клика мыши
def mouse_callback(event, x, y, flags, param):
    global rectangles

    if event == cv2.EVENT_LBUTTONDOWN:
        top_left = (x - RECT_SIZE // 2, y - RECT_SIZE // 2)
        bottom_right = (x + RECT_SIZE // 2, y + RECT_SIZE // 2)
        rectangles.append((top_left, bottom_right))

# Основная программа
def main():
    global rectangles

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: 0 for webcam or path to video file"
    )
    args = parser.parse_args()

    # Определяем источник видео
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print("Ошибка открытия видео")
        return

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Рисуем все прямоугольники
        for rect in rectangles:
            cv2.rectangle(frame, rect[0], rect[1], (255, 0, 255), 2)

        cv2.imshow("Video", frame)

        key = cv2.waitKey(1) & 0xFF

        # Очистка по C
        if key == ord('c') or key == ord('C'):
            rectangles.clear()

        # Выход по Q
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
