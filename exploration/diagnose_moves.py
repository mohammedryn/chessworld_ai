"""Phase-1 diagnostic: WHY does each move fire?

Runs the REAL pipeline (board detect -> warp -> YOLO -> state machine) but
injects an instrumented BoardStateMachine that records, for every emitted move:
  - source/dest squares (row,col + algebraic)
  - whether source or dest sits on the board edge (outer ring)
  - whether the move is an adjacent-square "jitter" (|drow|<=1 and |dcol|<=1)
  - the occupancy match score the move achieved (out of 64)
  - which squares changed between the committed board and the voted board

Touches NO source files: it patches src.pipeline.BoardStateMachine at runtime.

Usage:
    python exploration/diagnose_moves.py --video videos/chess3.mp4
    python exploration/diagnose_moves.py --video videos/game3.mp4   # control
"""
import sys, argparse, tempfile
from pathlib import Path

sys.path.insert(0, 'D:/chessworldai_assignment/chessvision-pgn')

import chess
import src.pipeline as pipeline_mod
from src.state_machine import BoardStateMachine

EVENTS = []  # collected per-move diagnostics


def _alg(row, col):
    """row,col in detected coords (row0=top, flipped=False) -> algebraic square."""
    # mirrors _sq_to_rc inverse for flipped=False: sq = chess.square(col, 7-row)
    return chess.square_name(chess.square(col, 7 - row))


def _is_edge(row, col):
    return row in (0, 7) or col in (0, 7)


class InstrumentedSM(BoardStateMachine):
    def update(self, board_state, frame_idx=0):
        committed_before = None if self._committed is None else self._committed.copy()
        board_before = self._chess_board.copy()
        move = super().update(board_state, frame_idx=frame_idx)
        if move is not None and committed_before is not None:
            fr, fc = self._sq_to_rc(move.from_square)
            tr, tc = self._sq_to_rc(move.to_square)
            test = board_before.copy()
            test.push(move)
            score = self._match_score(test, self._committed)
            # what was at the source square in the committed-before board?
            src_before = committed_before[fr][fc]
            src_code = src_before[0] if src_before is not None else None
            changed = []
            for r in range(8):
                for c in range(8):
                    a = committed_before[r][c][0] if committed_before[r][c] is not None else None
                    b = self._committed[r][c][0] if self._committed[r][c] is not None else None
                    if a != b:
                        changed.append((r, c, a, b))
            EVENTS.append({
                "frame": frame_idx,
                "san": board_before.san(move),
                "uci": move.uci(),
                "from_rc": (fr, fc), "to_rc": (tr, tc),
                "from_sq": _alg(fr, fc), "to_sq": _alg(tr, tc),
                "from_edge": _is_edge(fr, fc), "to_edge": _is_edge(tr, tc),
                "adjacent": abs(fr - tr) <= 1 and abs(fc - tc) <= 1,
                "src_code_committed": src_code,
                "score": score,
                "n_changed": len(changed),
                "changed": changed,
            })
        return move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--min-frame-gap", type=int, default=60)
    ap.add_argument("--move-threshold", type=int, default=32)
    args = ap.parse_args()

    pipeline_mod.BoardStateMachine = InstrumentedSM  # inject

    pipe = pipeline_mod.ChessVisionPipeline(
        piece_model_path="models/piece_detector.pt", log_path=None, confidence=0.25,
    )
    tmp_out = Path(tempfile.gettempdir()) / (Path(args.video).stem + "_diag.pgn")
    pipe.process_video(
        args.video, str(tmp_out),
        min_frame_gap=args.min_frame_gap, move_threshold=args.move_threshold,
    )
    pipe.close()

    n = len(EVENTS)
    print("\n" + "=" * 70)
    print(f"DIAGNOSTIC SUMMARY for {Path(args.video).name}  ({n} moves emitted)")
    print("=" * 70)
    if n == 0:
        return
    edge_moves = sum(1 for e in EVENTS if e["from_edge"] or e["to_edge"])
    adj_moves = sum(1 for e in EVENTS if e["adjacent"])
    both_edge = sum(1 for e in EVENTS if e["from_edge"] and e["to_edge"])
    scores = sorted(e["score"] for e in EVENTS)
    print(f"  moves touching board edge (outer ring): {edge_moves}/{n} = {edge_moves/n:.0%}")
    print(f"  moves with BOTH src & dst on edge:       {both_edge}/{n} = {both_edge/n:.0%}")
    print(f"  adjacent-square moves (|d|<=1):          {adj_moves}/{n} = {adj_moves/n:.0%}")
    print(f"  match score: min={scores[0]} median={scores[n//2]} max={scores[-1]} (threshold=32)")
    print(f"  scores <=40: {sum(1 for s in scores if s <= 40)}/{n}")
    print("\n  per-move detail:")
    print(f"  {'#':>2} {'san':<7} {'from->to':<9} {'edge':<10} {'adj':<4} {'score':>5} {'nChg':>4}")
    for i, e in enumerate(EVENTS, 1):
        edge = ("S" if e["from_edge"] else "-") + ("D" if e["to_edge"] else "-")
        print(f"  {i:>2} {e['san']:<7} {e['from_sq']}->{e['to_sq']:<5} "
              f"{edge:<10} {str(e['adjacent']):<5} {e['score']:>5} {e['n_changed']:>4}")


if __name__ == "__main__":
    main()
