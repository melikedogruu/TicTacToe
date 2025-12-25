# =========================
# FOR TESTING BOARD IMAGE TO CELLS EXTRACTION
# =========================

import cv2
import numpy as np

# =========================
# LOAD IMAGE
# =========================
img = cv2.imread("board_test.jpeg")
orig = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# =========================
# EDGE DETECTION
# =========================
edges = cv2.Canny(blur, 50, 150)

# =========================
# FIND CONTOURS OF OUTER BOARD
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

# =========================
# PERSPECTIVE TRANSFORM
# =========================
pts = board_contour.reshape(4,2)

# Order points: top-left, top-right, bottom-right, bottom-left
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
    [maxWidth-1, 0],
    [maxWidth-1, maxHeight-1],
    [0, maxHeight-1]
], dtype="float32")

M = cv2.getPerspectiveTransform(rect, dst)
warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

cv2.imshow("Warped Board", warp)
cv2.waitKey(0)

# =========================
# SPLIT INTO 9 CELLS
# =========================
cell_w = maxWidth // 3
cell_h = maxHeight // 3

cells = []

for row in range(3):
    for col in range(3):
        x1 = col * cell_w
        y1 = row * cell_h
        x2 = (col+1) * cell_w
        y2 = (row+1) * cell_h

        cell = warp[y1:y2, x1:x2]
        cells.append(cell)

        cv2.imshow(f"Cell {row*3 + col + 1}", cell)

cv2.waitKey(0)
cv2.destroyAllWindows()