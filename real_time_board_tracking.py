import cv2
import numpy as np
import tensorflow as tf
import time
from collections import defaultdict, deque

# =========================
# Robot Cell Mapping (Vision to Robot)
# =========================
ROBOT_GRID_MAP = {
    (0, 0): 1,
    (0, 1): 2,
    (0, 2): 3,
    (1, 0): 4,
    (1, 1): 5,
    (1, 2): 6,
    (2, 0): 7,
    (2, 1): 8,
    (2, 2): 9
}

cell_history = defaultdict(list)
last_process_time = 0
PROCESS_INTERVAL = 0.4   
from game import Game

cv2.namedWindow("Live Camera Feed", cv2.WINDOW_NORMAL)
cv2.namedWindow("Live Board with State", cv2.WINDOW_NORMAL)

def check_is_over(g: Game, piece: str):
    player_turn = 1 if piece == "X" else -1
    isover = g.is_over()
    if isover is None:
        return False
    if isover == 0:
        print("Tie")
        return True
    if isover == player_turn:
        print("You Win")
    else:
        print("You Lose")
    return True


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
PHONE_CAMERA_IP = "http://10.249.155.67:8080/video?dummy=param.mjpg"

cap = cv2.VideoCapture(PHONE_CAMERA_IP, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Phone camera not connected!")
    exit()

previous_board = [["empty" for _ in range(3)] for _ in range(3)]

# =========================
# Difficulty Level and Play Order Selection
# =========================
diff = input("Select Difficulty:\n0: Easy (random)\n1:Eh\n2:Medium\n3:Hard")
while (diff not in ["0","1","2","3"]):
    print("Not a valid input")
    diff = input("Select Difficulty:\n0: Easy (random)\n1:Eh\n2:Medium\n3:Hard\n")
g = Game(int(diff))

player = input("Play order:\n1-First Player\n2-Second Player\n")
while (not player.isdecimal() or not(1 <= int(player) <= 2)):
    print("Not a valid input")
    player = input("Play order:\n1-First Player\n2-Second Player\n")
player = int(player)

# =========================
# Initialize piece variables for demo human will be O and bot X
# =========================
human_piece = "O"
bot_piece = "X"
piece = human_piece


if player == 1:
    # Human (O) starts
    g.turn = -1
    print("Human starts as O. Robot will respond as X.")
else:
    # Robot (X) starts
    g.turn = 1
    print("Robot starts as X.")
    move = g.bot_move()
    if not move:
        print("No bot move returned. Game cannot start.")
        exit()
    rV, cV = move
    grid_ID = ROBOT_GRID_MAP[(rV, cV)]
    print(f"Bot Move: [{rV},{cV}], to grid number {grid_ID}")
    # TODO: send grid_ID to actuation 
    

accepted_human_moves = set()
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
    changes = []
    for rr in range(3):
        for cc in range(3):
            if board_matrix[rr][cc] != previous_board[rr][cc]:
                changes.append((rr, cc, previous_board[rr][cc], board_matrix[rr][cc]))

    # accept only one NEW mark (X or O) per update
    valid_changes = [(rr, cc, old, new) for (rr, cc, old, new) in changes if new in ["X", "O"]]

    if len(valid_changes) == 1:
        row, col, old, new = valid_changes[0]

        # ---- stability check ----
        key = (row, col)
        if cell_history[key].count(new) < 5:
            continue

        # ---- ignore repeated O detections (flicker protection) ----
        if new == "O" and (row, col) in accepted_human_moves:
            print(f"Ignoring repeated O at V[{row},{col}]")
            continue

        print("\n✅ BOARD UPDATED:")
        for rr in board_matrix:
            print(rr)

        # update snapshot AFTER we accept stability
        previous_board = [rr.copy() for rr in board_matrix]

        # Only accept human O moves from vision (ignore robot's X detections)
        if new != "O":
            print(f"Ignoring detected {new} at V[{row},{col}] (robot's mark)")
            continue

        # ---- apply human move to Game ----
        ok = g.move("O", row, col)
        if not ok:
            print(f"Game rejected O at V[{row},{col}] (turn={g.turn}).")
            continue

        accepted_human_moves.add((row, col))  # lock this human move

        # ---- check game end after human ----
        res = g.is_over()
        if check_is_over(g, human_piece):
            break

        # ---- bot plays X ----
        move = g.bot_move()

        # move must be exactly [r, c]
        if not isinstance(move, (list, tuple)) or len(move) != 2:
            print("Bot move invalid / game over:", move)
            break

        rV, cV = move
        grid_ID = ROBOT_GRID_MAP[(rV, cV)]
        print(f"Bot Move: [{rV},{cV}], to grid number {grid_ID}")
        # TODO: send grid_ID to actuation here

        # ---- check game end after bot ----
        res = g.is_over()
        if check_is_over(g, bot_piece):
            break

    # else: no valid single move detected -> do nothing, keep showing frames

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