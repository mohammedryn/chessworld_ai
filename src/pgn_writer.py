import io
import chess
import chess.pgn
from datetime import date
from typing import Optional


class PGNWriter:
    def __init__(
        self,
        white: str = "?",
        black: str = "?",
        event: str = "ChessWorld AI Assignment",
    ):
        self._game = chess.pgn.Game()
        self._game.headers["Event"] = event
        self._game.headers["Date"] = date.today().strftime("%Y.%m.%d")
        self._game.headers["White"] = white
        self._game.headers["Black"] = black
        self._game.headers["Result"] = "*"
        self._node = self._game

    def add_move(self, move: chess.Move, comment: Optional[str] = None):
        self._node = self._node.add_variation(move)
        if comment:
            self._node.comment = comment

    def to_string(self) -> str:
        buf = io.StringIO()
        self._game.accept(chess.pgn.FileExporter(buf))
        return buf.getvalue()

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_string())

    def move_count(self) -> int:
        return self._game.end().ply()
