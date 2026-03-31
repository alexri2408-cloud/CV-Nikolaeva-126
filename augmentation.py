import cv2
import numpy as np
import os
import random

# Путь к датасету
DATASET_PATH = r"D:\CVLabs\Lab4\mydataset"

# Получаем список изображений
images = [f for f in os.listdir(DATASET_PATH) if f.endswith(".png")]

def load_image(name):
    return cv2.imread(os.path.join(DATASET_PATH, name))

def save_image(name, suffix, img):
    base, ext = os.path.splitext(name)
    new_name = f"{base}_{suffix}{ext}"
    cv2.imwrite(os.path.join(DATASET_PATH, new_name), img)

# -----------------------------
# 1) GRAYSCALE
# -----------------------------
for img_name in random.sample(images, 13):
    img = load_image(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_image(img_name, "togray", gray)

# -----------------------------
# 2) RESIZE_HALF
# -----------------------------
for img_name in random.sample(images, 13):
    img = load_image(img_name)
    half = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    save_image(img_name, "half", half)

# -----------------------------
# 3) RESIZE_BIGGER
# -----------------------------
for img_name in random.sample(images, 13):
    img = load_image(img_name)
    bigger = cv2.resize(img, (2000, 3000))
    save_image(img_name, "bigger", bigger)

# -----------------------------
# 4) CROP
# -----------------------------
for img_name in random.sample(images, 13):
    img = load_image(img_name)
    crop = img[360:720, 640:1280]
    save_image(img_name, "crop", crop)

# -----------------------------
# 5) FLIP_VERT
# -----------------------------
for img_name in random.sample(images, 13):
    img = load_image(img_name)
    flip_vert = cv2.flip(img, 0)  # вертикальное отражение
    save_image(img_name, "flip_vert", flip_vert)

# -----------------------------
# 6) FLIP_HOR
# -----------------------------
for img_name in random.sample(images, 13):
    img = load_image(img_name)
    flip_hor = cv2.flip(img, 1)  # горизонтальное отражение
    save_image(img_name, "flip_hor", flip_hor)

# -----------------------------
# 7) ROTATE_90CLOCKWISE
# -----------------------------
for img_name in random.sample(images, 13):
    img = load_image(img_name)
    rot_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    save_image(img_name, "rot_90_clockwise", rot_90_clockwise)

# -----------------------------
# 8) ROTATE_90COUNTERCLOCKWISE
# -----------------------------
for img_name in random.sample(images, 13):
    img = load_image(img_name)
    rot_90_counterclockwise = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    save_image(img_name, "rot_90_counterclockwise", rot_90_counterclockwise)

# -----------------------------
# 9) ROTATE_180
# -----------------------------
for img_name in random.sample(images, 13):
    img = load_image(img_name)
    rot_180 = cv2.rotate(img, cv2.ROTATE_180)
    save_image(img_name, "rot_180", rot_180)

# -----------------------------
# 10) ROTATE_45
# -----------------------------
for img_name in random.sample(images, 13):
    img = load_image(img_name)
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
    rot_45 = cv2.warpAffine(img, M, (w, h))
    save_image(img_name, "rot_45", rot_45)

# -----------------------------
# 11) BLUR
# -----------------------------
for img_name in random.sample(images, 13):
    img = load_image(img_name)
    blurred = cv2.GaussianBlur(img, (29, 29), 0)
    save_image(img_name, "blur", blurred)

# -----------------------------
# 12) BRIGHTNESS
# -----------------------------
for img_name in random.sample(images, 13):
    img = load_image(img_name)
    adjusted = cv2.convertScaleAbs(img, alpha=1.5, beta=10)
    save_image(img_name, "bright", adjusted)

# -----------------------------
# 13) NOISE
# -----------------------------
def add_noise(image, mean=0, std=5):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

for img_name in random.sample(images, 13):
    img = load_image(img_name)
    noisy = add_noise(img)
    save_image(img_name, "noise", noisy)

print("Аугментация завершена!")
