import cv2
import numpy as np

def estimate_angle_from_bbox(bbox):
    """
    Оценка угла наклона QR кода по верхнему ребру
    bbox: массив 4 точек
    """
    if bbox is None:
        return None

    pts = bbox[0]

    # верхняя грань (между первой и второй точкой)
    pt1 = pts[0]
    pt2 = pts[1]

    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]

    angle = np.degrees(np.arctan2(dy, dx))
    return angle


def main():
    cap = cv2.VideoCapture(0)  # камера ноутбука

    detector = cv2.QRCodeDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data, bbox, _ = detector.detectAndDecode(frame)

        if bbox is not None:
            # рисуем рамку
            for i in range(len(bbox[0])):
                pt1 = tuple(bbox[0][i].astype(int))
                pt2 = tuple(bbox[0][(i + 1) % len(bbox[0])].astype(int))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

            # вычисляем угол
            angle = estimate_angle_from_bbox(bbox)

            if angle is not None:
                cv2.putText(frame, f"ANGLE: {angle:.2f}",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

        # вывод данных QR
        if data:
            cv2.putText(frame, f"DATA: {data}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        # режим RAW
        cv2.putText(frame, "Mode: RAW",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

        cv2.imshow("QR Scanner", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
