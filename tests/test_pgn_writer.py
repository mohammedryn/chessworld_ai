import chess
import chess.pgn
import io
import pytest
from src.pgn_writer import PGNWriter


def test_empty_pgn_is_parseable():
    writer = PGNWriter()
    pgn_str = writer.to_string()
    game = chess.pgn.read_game(io.StringIO(pgn_str))
    assert game is not None


def test_pgn_headers_present():
    writer = PGNWriter(white="Alice", black="Bob", event="Test Event")
    pgn_str = writer.to_string()
    assert "Alice" in pgn_str
    assert "Bob" in pgn_str
    assert "Test Event" in pgn_str


def test_add_moves_and_parse():
    writer = PGNWriter()
    moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("e7e5"),
        chess.Move.from_uci("g1f3"),
    ]
    for move in moves:
        writer.add_move(move)

    pgn_str = writer.to_string()
    game = chess.pgn.read_game(io.StringIO(pgn_str))
    assert game is not None
    node = game
    parsed_moves = []
    while node.variations:
        node = node.variations[0]
        parsed_moves.append(node.move)
    assert len(parsed_moves) == 3
    assert parsed_moves[0] == chess.Move.from_uci("e2e4")


def test_move_count():
    writer = PGNWriter()
    for uci in ["e2e4", "e7e5", "d2d4"]:
        writer.add_move(chess.Move.from_uci(uci))
    assert writer.move_count() == 3


def test_comment_appears_in_pgn():
    writer = PGNWriter()
    writer.add_move(chess.Move.from_uci("e2e4"), comment="promoted to Queen by default")
    pgn_str = writer.to_string()
    assert "promoted to Queen by default" in pgn_str


def test_save_writes_file(tmp_path):
    writer = PGNWriter()
    writer.add_move(chess.Move.from_uci("e2e4"))
    path = tmp_path / "game.pgn"
    writer.save(str(path))
    assert path.exists()
    content = path.read_text()
    assert "e4" in content
