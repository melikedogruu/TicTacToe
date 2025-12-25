import numpy as np
import copy
import math
from typing import List, Tuple, Optional


NUM_OF_COLS = 3
NUM_OF_ROWS = 3

CHARS = {
    None: "   ",
    1: " X ",
    -1: " O "
}


class GameError(Exception):
    def __init__(self, message: str, exc_type: str):
        super().__init__(message)
        self.type = exc_type

    def to_string(self) -> str:
        base = super().__str__()
        return f"Error {self.type} | {base}"


class Game:
    def __init__(self, difficulty: int):
        self.board: List[List[Optional[int]]] = [[None for _ in range(3)] for _ in range(3)]
        self.vision_matrix: List[List[Optional[int]]] = [[None for _ in range(3)] for _ in range(3)]
        self.moves: List[List[int]] = []
        self.diff = difficulty
        self.game_over = False
        self.turn = 1  # 1 = X, -1 = O

    # ---------------------------
    # Display
    # ---------------------------
    def draw_board(self) -> None:
        print()
        for row in range(3):
            line = []
            for col in range(3):
                line.append(CHARS[self.board[row][col]])
            print("%s" % ("|".join(line)))
            if row != 2:
                print("------------")

    # ---------------------------
    # Move generation + win checks
    # ---------------------------
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        moves: List[Tuple[int, int]] = []
        for i in range(NUM_OF_ROWS):
            for j in range(NUM_OF_COLS):
                if self.board[i][j] is None:
                    moves.append((i, j))
        return moves

    def has_winner(self) -> Optional[int]:
        # rows
        for row in range(3):
            unique = set(self.board[row])
            if len(unique) == 1:
                val = next(iter(unique))
                if val is not None:
                    return val

        # cols
        for col in range(3):
            unique = {self.board[row][col] for row in range(3)}
            if len(unique) == 1:
                val = next(iter(unique))
                if val is not None:
                    return val

        # diag TL->BR
        diag1 = {self.board[0][0], self.board[1][1], self.board[2][2]}
        if len(diag1) == 1:
            val = next(iter(diag1))
            if val is not None:
                return val

        # diag BL->TR
        diag2 = {self.board[2][0], self.board[1][1], self.board[0][2]}
        if len(diag2) == 1:
            val = next(iter(diag2))
            if val is not None:
                return val

        return None

    def is_over(self) -> Optional[int]:
        winner = self.has_winner()
        if winner is not None:
            self.game_over = True
            return winner

        if len(self.get_legal_moves()) == 0:
            self.game_over = True
            return 0  # tie

        return None

    # ---------------------------
    # Making a move
    # ---------------------------
    def move(self, piece: str, row: int, col: int) -> bool:
        try:
            if (self.turn == 1 and piece == "O") or (self.turn == -1 and piece == "X"):
                raise GameError("Two moves in succession were played", "ILLEGAL MOVE")

            if (row, col) not in self.get_legal_moves():
                raise GameError(f"The cell ({row}, {col}) is not empty or out of range", "ILLEGAL MOVE")

            self.board[row][col] = self.turn
            self.moves.append([row, col])
            self.turn *= -1
            return True

        except GameError:
            return False

    # ---------------------------
    # Bot logic wrappers
    # ---------------------------
    def bot_move(self) -> list[int]:
        # You can replace this print with your OpenCV vision_matrix integration later.
        # print(self.vision_matrix)

        if self.diff == 0:
            self.random_move()
        elif self.diff == 1:
            self.one_layer()
        elif self.diff == 2:
            self.two_layer()
        elif self.diff == 3:
            self.minimaxController()
        else:
            self.random_move()

    def random_move(self) -> list[int]:
        legal = self.get_legal_moves()
        if not legal:
            return []
        r, c = legal[np.random.randint(len(legal))]
        expected = "X" if self.turn == 1 else "O"
        self.move(expected, r, c)
        return [r, c]

    def one_layer(self) -> None:
        candidates = self.get_legal_moves()
        expected = "X" if self.turn == 1 else "O"
        me = self.turn  # 1 or -1

        # If any move wins immediately, take it; else random
        for (r, c) in candidates:
            ng = self.copy()
            ng.board[r][c] = me
            if ng.has_winner() == me:
                self.move(expected, r, c)
                return [r, c]

        return self.random_move()

    def two_layer(self) -> list[int]:
        candidates = self.get_legal_moves()
        expected = "X" if self.turn == 1 else "O"
        me = self.turn
        opp = -self.turn

        # 1) if I can win now, do it
        for (r, c) in candidates:
            ng = self.copy()
            ng.board[r][c] = me
            if ng.has_winner() == me:
                self.move(expected, r, c)
                return [r, c]

        # 2) if opponent can win next, block it
        for (r, c) in candidates:
            ng = self.copy()
            ng.board[r][c] = me  # assume I play (r,c)
            # check if opponent has an immediate winning reply anywhere
            for (rr, cc) in ng.get_legal_moves():
                ng2 = ng.copy()
                ng2.board[rr][cc] = opp
                if ng2.has_winner() == opp:
                    # block opponent's winning square instead (rr,cc) in the REAL board
                    self.move(expected, rr, cc)
                    return [r, c]

        # 3) otherwise random
        return self.random_move()

    # ---------------------------
    # Minimax (alpha-beta)
    # ---------------------------
    def minimax(self, alpha: float, beta: float, is_maximizing: bool, max_player: int) -> float:
        w = self.has_winner()
        if w is not None:
            return 1.0 if w == max_player else -1.0

        if all(self.board[i][j] is not None for i in range(3) for j in range(3)):
            return 0.0

        min_player = -max_player

        if is_maximizing:
            best = -math.inf
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] is None:
                        self.board[i][j] = max_player
                        best = max(best, self.minimax(alpha, beta, False, max_player))
                        self.board[i][j] = None
                        alpha = max(alpha, best)
                        if alpha >= beta:
                            return best
            return best
        else:
            best = math.inf
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] is None:
                        self.board[i][j] = min_player
                        best = min(best, self.minimax(alpha, beta, True, max_player))
                        self.board[i][j] = None
                        beta = min(beta, best)
                        if beta <= alpha:
                            return best
            return best

    def minimaxController(self) -> list[int]:
        expected = "X" if self.turn == 1 else "O"
        max_player = self.turn  # 1 or -1

        # small optimization: take center if free
        if self.board[1][1] is None:
            self.move(expected, 1, 1)
            return [1,1]

        best_score = -math.inf
        best_move = (-1, -1)

        for i in range(3):
            for j in range(3):
                if self.board[i][j] is None:
                    self.board[i][j] = max_player
                    score = self.minimax(-math.inf, math.inf, False, max_player)
                    self.board[i][j] = None

                    if score > best_score:
                        best_score = score
                        best_move = (i, j)

        if best_move != (-1, -1):
            self.move(expected, best_move[0], best_move[1])
            return [best_move[0], best_move[1]]

    # ---------------------------
    # CLI play loop
    # ---------------------------
    def play(self) -> None:
        player = input("Play as:\n1- X\n2- O\nPicking O will let you play 2nd\n").strip()
        while (not player.isdecimal()) or (int(player) not in (1, 2)):
            print("Not a valid input")
            player = input("Play as:\n1- X\n2- O\nPicking O will let you play 2nd\n").strip()

        human_piece = "X" if player == "1" else "O"

        # if human chose O, bot starts
        if human_piece == "O":
            self.bot_move()

        while not self.game_over:
            self.draw_board()

            raw = input("Enter move as 'ROW COL' (0-2 0-2): ").strip().split()
            if len(raw) != 2 or (not raw[0].isdigit()) or (not raw[1].isdigit()):
                print("Invalid format. Example: 1 2")
                continue

            r, c = int(raw[0]), int(raw[1])
            if not (0 <= r <= 2 and 0 <= c <= 2):
                print("Row/Col must be between 0 and 2.")
                continue

            ok = self.move(human_piece, r, c)
            if not ok:
                print("Illegal move. Try again.")
                continue

            result = self.is_over()
            if result == 1:
                print("You Win!\n")
                self.draw_board()
                return
            if result == -1:
                print("You Lose!\n")
                self.draw_board()
                return
            if result == 0:
                print("Tie!\n")
                self.draw_board()
                return

            # bot turn
            self.bot_move()
            result = self.is_over()
            if result == 1:
                # X won
                if human_piece == "X":
                    print("You Win!\n")
                else:
                    print("You Lose!\n")
                self.draw_board()
                return
            if result == -1:
                # O won
                if human_piece == "O":
                    print("You Win!\n")
                else:
                    print("You Lose!\n")
                self.draw_board()
                return
            if result == 0:
                print("Tie!\n")
                self.draw_board()
                return

    # ---------------------------
    # Copy
    # ---------------------------
    def copy(self) -> "Game":
        target = Game(self.diff)
        target.board = copy.deepcopy(self.board)
        target.moves = copy.deepcopy(self.moves)
        target.turn = self.turn
        target.game_over = self.game_over
        return target


def main() -> None:
    g = Game(3)  # 0 random, 1 one-layer, 2 two-layer, 3 minimax
    g.play()


if __name__ == "__main__":
    main()