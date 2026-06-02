import argparse
import json
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


ROOM_TEMPLATES = [
    [
        "#######",
        "#     #",
        "# ### #",
        "#     #",
        "# ### #",
        "#     #",
        "#######",
    ],
    [
        "########",
        "#      #",
        "# ## # #",
        "#      #",
        "# # ## #",
        "#      #",
        "########",
    ],
    [
        "########",
        "#      #",
        "# #### #",
        "#      #",
        "# #### #",
        "#      #",
        "########",
    ],
    [
        "########",
        "#      #",
        "#  ##  #",
        "#      #",
        "#  ##  #",
        "#      #",
        "########",
    ],
    [
        "#########",
        "#       #",
        "# ### # #",
        "#       #",
        "# # ### #",
        "#       #",
        "#########",
    ],
]


PROMPT_TEMPLATE = """<image>
你正在求解一个 Sokoban（推箱子）状态。图像展示当前盘面，下面也给出文字版网格。

符号说明：
- `#` 墙
- `_` 空地
- `O` 目标点
- `X` 箱子
- `P` 玩家
- `√` 目标点上的箱子
- `S` 目标点上的玩家

任务：只输出当前这一步的思考和动作，不要一次性输出整条轨迹。

当前是轨迹 `{trajectory_id}` 的第 `{step_idx}` / `{total_steps}` 步。
当前观测：
{board_text}

最近动作历史：{history_text}

请先在 `<think>...</think>` 中给出简短、合理的规则推理，再在 `<action>...</action>` 中只输出一个动作，可选值为 `up` / `down` / `left` / `right`。"""


@dataclass(frozen=True)
class Puzzle:
    width: int
    height: int
    walls: frozenset[tuple[int, int]]
    targets: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class State:
    player: tuple[int, int]
    boxes: tuple[tuple[int, int], ...]


def add_pos(pos: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
    return pos[0] + delta[0], pos[1] + delta[1]


def parse_room(template: list[str]) -> tuple[int, int, frozenset[tuple[int, int]], list[tuple[int, int]]]:
    walls = set()
    floors = []
    height = len(template)
    width = len(template[0])
    for y, row in enumerate(template):
        for x, cell in enumerate(row):
            if cell == "#":
                walls.add((x, y))
            else:
                floors.append((x, y))
    return width, height, frozenset(walls), floors


def sorted_boxes(boxes: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(boxes))


def is_solved(state: State, puzzle: Puzzle) -> bool:
    return set(state.boxes) == set(puzzle.targets)


def apply_forward_action(state: State, puzzle: Puzzle, action: str) -> State | None:
    delta = DIRECTIONS[action]
    next_player = add_pos(state.player, delta)
    boxes = set(state.boxes)
    if next_player in puzzle.walls:
        return None
    if next_player in boxes:
        pushed_box = add_pos(next_player, delta)
        if pushed_box in puzzle.walls or pushed_box in boxes:
            return None
        boxes.remove(next_player)
        boxes.add(pushed_box)
        return State(player=next_player, boxes=sorted_boxes(boxes))
    return State(player=next_player, boxes=state.boxes)


def reverse_action_options(state: State, puzzle: Puzzle) -> list[State]:
    boxes = set(state.boxes)
    candidates = []
    for delta in DIRECTIONS.values():
        prev_player = add_pos(state.player, (-delta[0], -delta[1]))
        if prev_player not in puzzle.walls and prev_player not in boxes:
            candidates.append(State(player=prev_player, boxes=state.boxes))
        box_pos = add_pos(state.player, delta)
        behind_player = add_pos(state.player, (-delta[0], -delta[1]))
        if box_pos in boxes and behind_player not in puzzle.walls and behind_player not in boxes:
            new_boxes = set(boxes)
            new_boxes.remove(box_pos)
            new_boxes.add(state.player)
            candidates.append(State(player=behind_player, boxes=sorted_boxes(new_boxes)))
    return candidates


def shortest_solution(initial_state: State, puzzle: Puzzle, max_expansions: int = 50000) -> list[str] | None:
    from collections import deque

    queue = deque([(initial_state, [])])
    visited = {initial_state}
    expansions = 0

    while queue:
        state, path = queue.popleft()
        expansions += 1
        if expansions > max_expansions:
            return None
        if is_solved(state, puzzle):
            return path
        for action in DIRECTIONS:
            next_state = apply_forward_action(state, puzzle, action)
            if next_state is None or next_state in visited:
                continue
            visited.add(next_state)
            queue.append((next_state, path + [action]))
    return None


def sample_reachable_player(floors: list[tuple[int, int]], boxes: set[tuple[int, int]], rng: random.Random) -> tuple[int, int]:
    choices = [cell for cell in floors if cell not in boxes]
    return rng.choice(choices)


def generate_single_trajectory(trajectory_id: str, rng: random.Random, min_solution_len: int, max_solution_len: int) -> dict:
    for _ in range(2000):
        template = rng.choice(ROOM_TEMPLATES)
        width, height, walls, floors = parse_room(template)
        num_boxes = 1 if len(floors) < 26 else rng.choice([1, 1, 2])
        targets = frozenset(rng.sample(floors, num_boxes))
        player = sample_reachable_player(floors, set(targets), rng)
        state = State(player=player, boxes=sorted_boxes(targets))
        puzzle = Puzzle(width=width, height=height, walls=walls, targets=targets)

        reverse_steps = rng.randint(max_solution_len, max_solution_len * 3)
        seen_scrambles = {state}
        box_move_count = 0
        for _ in range(reverse_steps):
            options = reverse_action_options(state, puzzle)
            if not options:
                break
            # Bias toward reverse-pulls so boxes actually move away from targets.
            options.sort(key=lambda candidate: candidate.boxes != state.boxes, reverse=True)
            top_k = min(4, len(options))
            next_state = rng.choice(options[:top_k])
            if next_state.boxes != state.boxes:
                box_move_count += 1
            state = next_state
            seen_scrambles.add(state)

        if box_move_count == 0 or is_solved(state, puzzle):
            continue

        solution = shortest_solution(state, puzzle)
        if solution is None:
            continue
        if not (min_solution_len <= len(solution) <= max_solution_len):
            continue

        return {
            "trajectory_id": trajectory_id,
            "puzzle": puzzle,
            "initial_state": state,
            "solution": solution,
            "template": template,
        }
    raise RuntimeError(f"Failed to generate a solvable puzzle for {trajectory_id}")


def replay_trajectory(initial_state: State, puzzle: Puzzle, solution: list[str]) -> list[State]:
    states = [initial_state]
    state = initial_state
    for action in solution:
        next_state = apply_forward_action(state, puzzle, action)
        if next_state is None:
            raise RuntimeError(f"Invalid action {action} during replay")
        states.append(next_state)
        state = next_state
    return states


def board_to_text(state: State, puzzle: Puzzle) -> str:
    rows = []
    boxes = set(state.boxes)
    for y in range(puzzle.height):
        row = []
        for x in range(puzzle.width):
            pos = (x, y)
            if pos in puzzle.walls:
                row.append("#")
            elif pos == state.player and pos in puzzle.targets:
                row.append("S")
            elif pos == state.player:
                row.append("P")
            elif pos in boxes and pos in puzzle.targets:
                row.append("√")
            elif pos in boxes:
                row.append("X")
            elif pos in puzzle.targets:
                row.append("O")
            else:
                row.append("_")
        rows.append(" ".join(row))
    return "\n".join(rows)


def render_state_image(state: State, puzzle: Puzzle, tile_size: int = 64) -> Image.Image:
    width_px = puzzle.width * tile_size
    height_px = puzzle.height * tile_size
    image = Image.new("RGB", (width_px, height_px), (238, 232, 220))
    draw = ImageDraw.Draw(image)
    boxes = set(state.boxes)

    for y in range(puzzle.height):
        for x in range(puzzle.width):
            left = x * tile_size
            top = y * tile_size
            right = left + tile_size
            bottom = top + tile_size
            pos = (x, y)

            if pos in puzzle.walls:
                draw.rectangle([left, top, right, bottom], fill=(59, 66, 82))
                draw.rectangle([left + 6, top + 6, right - 6, bottom - 6], outline=(94, 129, 172), width=3)
                continue

            draw.rectangle([left, top, right, bottom], fill=(240, 236, 226))
            draw.rectangle([left, top, right, bottom], outline=(210, 205, 194), width=1)

            if pos in puzzle.targets:
                draw.rounded_rectangle([left + 12, top + 12, right - 12, bottom - 12], radius=8, outline=(170, 52, 53), width=3, fill=(245, 229, 227))
                cx = (left + right) / 2
                cy = (top + bottom) / 2
                draw.polygon(
                    [
                        (cx, cy - 10),
                        (cx + 10, cy),
                        (cx, cy + 10),
                        (cx - 10, cy),
                    ],
                    fill=(188, 73, 74),
                )

            if pos in boxes:
                draw.rounded_rectangle([left + 10, top + 10, right - 10, bottom - 10], radius=6, fill=(222, 171, 73), outline=(169, 114, 18), width=3)
                draw.line([left + 18, top + 18, right - 18, bottom - 18], fill=(212, 95, 42), width=4)
                draw.line([right - 18, top + 18, left + 18, bottom - 18], fill=(212, 95, 42), width=4)

            if pos == state.player:
                cx = (left + right) / 2
                cy = (top + bottom) / 2 + 4
                draw.line([cx - 10, cy - 22, cx - 16, cy - 34], fill=(48, 122, 79), width=4)
                draw.line([cx + 10, cy - 22, cx + 16, cy - 34], fill=(48, 122, 79), width=4)
                draw.ellipse([cx - 20, cy - 24, cx + 20, cy + 18], fill=(77, 181, 103), outline=(48, 122, 79), width=3)
                draw.ellipse([cx - 10, cy - 8, cx - 3, cy - 1], fill=(20, 20, 20))
                draw.ellipse([cx + 3, cy - 8, cx + 10, cy - 1], fill=(20, 20, 20))

    return image


def encode_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def response_reasoning(state: State, next_state: State, action: str, remaining_actions: list[str]) -> str:
    pushed = state.boxes != next_state.boxes
    remaining = len(remaining_actions)
    if pushed:
        thought = (
            f"这一步要把箱子朝 `{action}` 方向推进，先让一个箱子继续沿可行路径接近目标点。"
            f"按规则搜索得到的最短剩余方案还有 {remaining} 步，所以当前动作应当保持这条路径。"
        )
    else:
        thought = (
            f"这一步还不能直接完成推箱，需要先移动到更合适的位置，为后续推箱创造站位。"
            f"按规则搜索得到的最短剩余方案还有 {remaining} 步，因此先执行 `{action}`。"
        )
    return f"<think>{thought}</think>\n<action>{action}</action>"


def build_prompt(trajectory_id: str, step_idx: int, total_steps: int, board_text: str, history: list[str]) -> str:
    history_text = "无" if not history else " -> ".join(history[-6:])
    return PROMPT_TEMPLATE.format(
        trajectory_id=trajectory_id,
        step_idx=step_idx,
        total_steps=total_steps,
        board_text=board_text,
        history_text=history_text,
    )


def trajectory_to_rows(trajectory: dict, output_dir: Path) -> tuple[list[dict], dict]:
    trajectory_id = trajectory["trajectory_id"]
    puzzle = trajectory["puzzle"]
    solution = trajectory["solution"]
    states = replay_trajectory(trajectory["initial_state"], puzzle, solution)
    image_dir = output_dir / "images" / trajectory_id
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    history = []
    for step_idx, action in enumerate(solution, start=1):
        state = states[step_idx - 1]
        next_state = states[step_idx]
        board_text = board_to_text(state, puzzle)
        prompt = build_prompt(trajectory_id, step_idx, len(solution), board_text, history)
        response = response_reasoning(state, next_state, action, solution[step_idx - 1 :])
        image = render_state_image(state, puzzle)
        image_bytes = encode_png(image)
        image_path = image_dir / f"step_{step_idx:02d}.png"
        image.save(image_path)

        rows.append(
            {
                "data_source": "sokoban_rule_sft",
                "trajectory_id": trajectory_id,
                "step_id": step_idx,
                "prompt": prompt,
                "response": response,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "images": [{"bytes": image_bytes}],
                "image_path": str(image_path.resolve()),
                "solution_actions": solution,
                "ability": "agent",
                "extra_info": {
                    "trajectory_id": trajectory_id,
                    "step_id": step_idx,
                    "solution_length": len(solution),
                    "next_action": action,
                },
            }
        )
        history.append(action)

    metadata = {
        "trajectory_id": trajectory_id,
        "solution_length": len(solution),
        "num_steps": len(solution),
        "actions": solution,
        "final_board": board_to_text(states[-1], puzzle),
    }
    return rows, metadata


def write_parquet(rows: list[dict], path: Path) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def generate_dataset(output_dir: Path, num_trajectories: int, seed: int, min_solution_len: int, max_solution_len: int) -> dict:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    trajectories = []
    used_signatures = set()
    while len(trajectories) < num_trajectories:
        trajectory_id = f"traj_{len(trajectories):03d}"
        trajectory = generate_single_trajectory(
            trajectory_id=trajectory_id,
            rng=rng,
            min_solution_len=min_solution_len,
            max_solution_len=max_solution_len,
        )
        signature = (tuple(sorted(trajectory["puzzle"].targets)), trajectory["initial_state"].player, trajectory["initial_state"].boxes)
        if signature in used_signatures:
            continue
        used_signatures.add(signature)
        trajectories.append(trajectory)

    train_cutoff = max(1, int(round(num_trajectories * 0.83)))
    train_rows, val_rows = [], []
    metadata = {"seed": seed, "num_trajectories": num_trajectories, "trajectories": []}

    for index, trajectory in enumerate(trajectories):
        rows, item_metadata = trajectory_to_rows(trajectory, output_dir)
        item_metadata["split"] = "train" if index < train_cutoff else "val"
        metadata["trajectories"].append(item_metadata)
        if index < train_cutoff:
            train_rows.extend(rows)
        else:
            val_rows.extend(rows)

    write_parquet(train_rows, output_dir / "train.parquet")
    write_parquet(val_rows, output_dir / "val.parquet")
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    readme = f"""# Sokoban Rule-Based SFT Dataset

This dataset was generated without calling any model.

- Trajectories: {num_trajectories}
- Train rows: {len(train_rows)}
- Val rows: {len(val_rows)}
- Seed: {seed}

Each parquet row contains:

- `prompt`: text prompt with `<image>` placeholder and ASCII board
- `response`: rule-based reasoning plus one correct action
- `messages`: user/assistant conversation form
- `images`: a list with one PNG stored as raw bytes
- `image_path`: the corresponding saved PNG on disk
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    return {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_trajectories": train_cutoff,
        "val_trajectories": num_trajectories - train_cutoff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small Sokoban SFT dataset with rule-based solutions.")
    parser.add_argument("--output-dir", default="data/sokoban_rule_sft", help="Directory used to store parquet files and rendered images.")
    parser.add_argument("--num-trajectories", type=int, default=12, help="Number of trajectories to generate.")
    parser.add_argument("--seed", type=int, default=20260408, help="Random seed.")
    parser.add_argument("--min-solution-len", type=int, default=4, help="Minimum accepted solution length.")
    parser.add_argument("--max-solution-len", type=int, default=12, help="Maximum accepted solution length.")
    args = parser.parse_args()

    summary = generate_dataset(
        output_dir=Path(args.output_dir),
        num_trajectories=args.num_trajectories,
        seed=args.seed,
        min_solution_len=args.min_solution_len,
        max_solution_len=args.max_solution_len,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
