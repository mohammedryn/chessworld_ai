import json
from pathlib import Path

def parse_pipeline_log():
    log_path = Path("D:/chessworldai_assignment/chessvision-pgn/pipeline.log")
    if not log_path.exists():
        print("Log file not found!")
        return

    moves = []
    occlusions = 0
    with open(log_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get("event") == "move":
                    moves.append(data)
                elif data.get("event") == "occlusion":
                    occlusions += 1
            except Exception as e:
                pass

    print(f"Total move events in log: {len(moves)}")
    print(f"Total occlusion events in log: {occlusions}")
    
    # Print the moves
    for idx, move in enumerate(moves):
        print(f"Move {idx+1}: Frame {move.get('frame')}, UCI: {move.get('uci')}")

if __name__ == "__main__":
    parse_pipeline_log()
