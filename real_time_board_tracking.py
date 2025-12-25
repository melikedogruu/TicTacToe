import cv2
import numpy as np
import tensorflow as tf
import time
from collections import defaultdict
cell_history = defaultdict(list)
last_process_time = 0
PROCESS_INTERVAL = 0.4   

from game import Game, GameError

cv2.namedWindow("Live Camera Feed", cv2.WINDOW_NORMAL)
cv2.namedWindow("Live Board with State", cv2.WINDOW_NORMAL)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("tic_tac_toe_cnn.h5")
class_names = ["O", "X", "empty"]

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_cell(cell_img):

    # SAFETY CHECK
    if cell_img is None or cell_img.size == 0:
        return np.zeros((1,128,128,3), dtype=np.float32)
    
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
# CONNECT TO PHONE CAMERA
# =========================
PHONE_CAMERA_IP = "http://10.247.62.9:8080/video?dummy=param.mjpg"

cap = cv2.VideoCapture(PHONE_CAMERA_IP, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Phone camera not connected")
    exit()

previous_board = None

# =========================
# REAL-TIME LOOP
# =========================
diff = int(input("Select Difficulty:\n0: Easy (random)\n1:Eh\n2:Medium\n3:Hard"))
g = Game(diff)
#TODO Valıd difficulty Check

player = input("Play as :\n1- X\n2- O\nPicking O will let you play 2nd\n")
while (not player.isdecimal() or not(1 <=int(player) <=2)):
    print("Not a valid input")
    player = input("Play as :\n1- X\n2- O\nPicking O will let you play 2nd\n")
piece = ""
if player == 1:
    piece = "X"
else:
    piece = "O"
    g.bot_move()


while True:
    ret, frame = cap.read()
    #print(ret) debug line
    current_time = time.time()
    if current_time - last_process_time < PROCESS_INTERVAL:
        cv2.imshow("Live Camera Feed", frame)
        if cv2.waitKey(1) == 27:
            break
        continue
    if not ret:
        print("❌ Frame not received")
        break

    original = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

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
        cv2.imshow("Live Board with State", frame)
        if cv2.waitKey(1) == 27:
            break
        continue


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
    warp = cv2.warpPerspective(original, M, (maxWidth, maxHeight))

    cell_w = maxWidth // 3
    cell_h = maxHeight // 3

    board_matrix = []

    # =========================
    # CLASSIFY 9 CELLS
    # =========================
    for row in range(3):
        row_vals = []
        for col in range(3):
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = (col+1) * cell_w
            y2 = (row+1) * cell_h

            cell = warp[y1:y2, x1:x2]

            # Safety check for empty crop
            if cell is None or cell.size == 0:
                row_vals.append("empty")
                continue

            processed = preprocess_cell(cell)
            pred = model.predict(processed, verbose=0)
            label = class_names[np.argmax(pred)]

            key = (row, col)
            cell_history[key].append(label)

            # Keep last 7 frames only
            if len(cell_history[key]) > 7:
                cell_history[key].pop(0)

            # Use most common label (majority vote)
            stable_label = max(set(cell_history[key]), key=cell_history[key].count)
            row_vals.append(stable_label)

        board_matrix.append(row_vals)
    last_process_time = time.time()

    # =========================
    # VISUALIZE DIGITAL BOARD
    # =========================
    vis = warp.copy()

    # Draw grid lines
    for i in range(1, 3):
        cv2.line(vis, (i*cell_w, 0), (i*cell_w, maxHeight), (0,0,255), 2)
        cv2.line(vis, (0, i*cell_h), (maxWidth, i*cell_h), (0,0,255), 2)

    # =========================
    # DRAW SYMBOLS WITH COLORS
    # =========================
    for r in range(3):
        for c in range(3):
            text = board_matrix[r][c]

            # Choose color
            if text == "X":
                color = (0, 0, 255)     # Red
            elif text == "O":
                color = (255, 0, 0)     # Blue
            else:
                color = (0, 255, 0) # Green for empty

            draw_text = text

            x = c * cell_w + cell_w // 2 - 40
            y = r * cell_h + cell_h // 2 + 20

            cv2.putText(
                vis,
                draw_text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,     
                color,
                3
            )
               
    # =========================
    # UPDATE PRINT ONLY IF STATE IS CHANGED
    # =========================
    if previous_board is None:
        previous_board = [row.copy() for row in board_matrix]
    difference_count = 0

    row, col = -1, -1
    for r in range(3):
        for c in range(3):
            if board_matrix[r][c] != previous_board[r][c]:
                difference_count += 1

    # Accept only REAL moves (1 new cell changes)
    if difference_count == 1:
        for r in range(3):
            for c in range(3):
                if previous_board[r][c] != board_matrix[r][c]:
                    row, col = r, c
                #print(board_matrix[r][c])
        print("\n✅ BOARD UPDATED:")
        for r in board_matrix:
            print(r)
        previous_board = [row.copy() for row in board_matrix]

        # Player Move then Bot move TODO

        if not g.game_over:
            g.move(piece, row, col)

            isover = g.is_over()
            if isover != None and isover != 0:
                print(f'You Win!\n')
                break
            elif isover == 0:
                print('Tie!')
                break

            r, c = g.bot_move()
            print(r, c)
            # TODO CHANNEL ROW COL DATA TO ARDUINO

            isover = g.is_over()
            if isover != None and isover != 0:
                print(f'You Win!\n')
                break
            elif isover == 0:
                print('Tie!')
                break

        else:
            break





    # =========================
    # SHOW WINDOWS
    # =========================
    cv2.imshow("Live Camera Feed", frame)
    cv2.imshow("Live Board with State", vis)

    if cv2.waitKey(1) == 27:
        break

cap.release()
time.sleep(1)
cap = cv2.VideoCapture(PHONE_CAMERA_IP, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cv2.waitKey(1)
cv2.destroyAllWindows()

"""
def play(self):
        player = input("Play as :\n1- X\n2- O\nPicking O will let you play 2nd\n")
        while (not player.isdecimal() or not(1 <=int(player) <=2)):
            print("Not a valid input")
            player = input("Play as :\n1- X\n2- O\nPicking O will let you play 2nd\n")
        piece = ""
        if player == "1":
            piece = "X"
        else:
            piece = "O"
        
        if player == "2":
            self.bot_move()
        while not self.game_over:
            self.draw_board()
            move = input("Enter move as 'ROW COL'").strip().split(" ")
            self.move(piece, int(move[0]), int(move[1]))
            print(isover)
            isover = self.is_over()
            if isover != None and isover != 0:
                print(f'You Win!\n')
                self.draw_board()
                return
            elif isover == 0:
                print('Tie!')
                self.draw_board()
                return
            self.bot_move()
            self.is_over()
            if isover != None and isover != 0:
                print(f'You Lose!\n')
                self.draw_board()
                return
            elif isover == 0:
                print('Tie!')
                self.draw_board()
                return
        

"""