
# =========================
# FOR TESTING BOARD IMAGE TO MATRIX CONVERSION
# =========================

import cv2
import numpy as np
import tensorflow as tf

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("tic_tac_toe_cnn.h5")


class_names = ["O", "X", "empty"]

# =========================
# PREPROCESS FUNCTION 
# =========================
def preprocess_cell(cell_img):
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        processed = np.zeros((128,128,3), dtype=np.float32)
        processed = np.expand_dims(processed, axis=0)
        return processed

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    crop = edges[y:y+h, x:x+w]

    side = max(w, h)
    square = np.zeros((side, side), dtype=np.uint8)
    square[:h, :w] = crop

    processed = cv2.resize(square, (128,128))
    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    processed = processed / 255.0
    processed = np.expand_dims(processed, axis=0)

    return processed

# =========================
# LOAD BOARD IMAGE
# =========================
img = cv2.imread("board_test.jpeg")
orig = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

edges = cv2.Canny(blur, 50, 150)

# =========================
# DETECT OUTER BOARD
# =========================
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

board_contour = None

for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        board_contour = approx
        break

if board_contour is None:
    print("❌ Board not detected")
    exit()

pts = board_contour.reshape(4,2)
rect = np.zeros((4,2), dtype="float32")

s = pts.sum(axis=1)
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]

diff = np.diff(pts, axis=1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]

(tl, tr, br, bl) = rect

widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
maxWidth = int(max(widthA, widthB))

heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
maxHeight = int(max(heightA, heightB))

dst = np.array([
    [0,0],
    [maxWidth-1,0],
    [maxWidth-1,maxHeight-1],
    [0,maxHeight-1]
], dtype="float32")

M = cv2.getPerspectiveTransform(rect, dst)
warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

# =========================
# SPLIT INTO 9 CELLS & CLASSIFY
# =========================
cell_w = maxWidth // 3
cell_h = maxHeight // 3

board_matrix = []

idx = 0
for row in range(3):
    row_vals = []
    for col in range(3):
        x1 = col * cell_w
        y1 = row * cell_h
        x2 = (col+1) * cell_w
        y2 = (row+1) * cell_h

        cell = warp[y1:y2, x1:x2]

        processed = preprocess_cell(cell)
        pred = model.predict(processed, verbose=0)
        label = class_names[np.argmax(pred)]

        row_vals.append(label)

    board_matrix.append(row_vals)

# =========================
# SHOW RESULTS
# =========================
print("\n✅ FINAL BOARD MATRIX:")
for row in board_matrix:
    print(row)
