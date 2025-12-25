# =========================
# PROCESS DATASET IMAGES FOR BETTER SYMBOL DETECTION
# =========================
import cv2
import os
import numpy as np

# ========= SETTINGS =========
INPUT_FOLDERS = ["train", "valid", "test"]
OUTPUT_SUFFIX = "_processed"
IMG_SIZE = 128

# ===========================
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    crop = edges[y:y+h, x:x+w]

    side = max(w, h)
    square = np.zeros((side, side), dtype=np.uint8)
    square[:h, :w] = crop

    resized = cv2.resize(square, (IMG_SIZE, IMG_SIZE))
    processed = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    return processed

# ===========================
for folder in INPUT_FOLDERS:
    input_base = folder
    output_base = folder + OUTPUT_SUFFIX

    os.makedirs(output_base, exist_ok=True)

    for class_name in os.listdir(input_base):
        input_class_path = os.path.join(input_base, class_name)
        output_class_path = os.path.join(output_base, class_name)

        os.makedirs(output_class_path, exist_ok=True)

        for fname in os.listdir(input_class_path):
            img_path = os.path.join(input_class_path, fname)
            img = cv2.imread(img_path)

            if img is None:
                continue

            proc = preprocess_image(img)
            save_path = os.path.join(output_class_path, fname)
            cv2.imwrite(save_path, proc)

    print(f"✅ Finished: {output_base}")

print("\n🎉 ALL DATASET IMAGES PREPROCESSED SUCCESSFULLY.")