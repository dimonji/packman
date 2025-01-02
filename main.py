import argparse
from collections import deque
from typing import List
from typing import Tuple

import numpy as np


def load_board(file_path: str) -> np.ndarray:
    """Loads the game board from a file."""
    # TODO: Add corner case handling:
    # - Check if the file exists
    # - Check if the file is in the correct format (npy)
    # - Check if the board has the correct shape (2D array)
    # - Check if the board has the correct values (0, 1, 2, 3)
    return np.load(file_path)


def find_coordinates(board: np.ndarray, value: int) -> List[Tuple[int, int]]:
    """Finds all coordinates of cells with the specified value."""
    return [(int(x), int(y)) for x, y in zip(*np.where(board == value))]


def bfs_with_obstacles(board: np.ndarray, start: Tuple[int, int], targets: List[Tuple[int, int]]) -> List[
    Tuple[Tuple[int, int], int]]:
    """BFS to analyze discrepancies."""
    rows, cols = board.shape
    visited = set()  # To keep track of visited cells
    queue = deque([(start, 0)])  # Queue of (position, distance)
    distances = {}  # To store distances to each target

    while queue:
        (x, y), dist = queue.popleft()

        # If this cell is a target, record its distance
        if (x, y) in targets:
            distances[(x, y)] = dist
            if len(distances) == len(targets):  # Stop if all targets are found
                break

        # Skip if already visited
        if (x, y) in visited:
            continue

        visited.add((x, y))

        # Explore neighbors (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy

            # Check boundaries and wall conditions
            if 0 <= nx < rows and 0 <= ny < cols and board[nx, ny] != 1 and (nx, ny) not in visited:
                queue.append(((nx, ny), dist + 1))

    # Return sorted list of distances to targets
    return [(target, distances.get(target, float('inf'))) for target in targets]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate distances from Pacman to ghosts.")
    parser.add_argument(
        "--board",
        type=str,
        required=True,
        help="Path to the board file (npy format)")
    args = parser.parse_args()

    # Load the game board
    board: np.ndarray = load_board(args.board)

    # Find the coordinates of Pacman and the ghosts
    pacman_pos: List[Tuple[int, int]] = find_coordinates(board, 3)

    if len(pacman_pos) != 1:
        raise ValueError("There must be exactly one Pacman on the board!")

    ghosts: List[Tuple[int, int]] = find_coordinates(board, 2)

    # Calculate distances from Pacman to each ghost
    distances: List[Tuple[Tuple[int, int], int]] = bfs_with_obstacles(board, pacman_pos[0], ghosts)
    sorted_distances = sorted(distances, key=lambda x: x[1])
    reachable_distances = [entry for entry in sorted_distances if entry[1] != float('inf')]

    # Print the result
    print(reachable_distances)


if __name__ == "__main__":
    main()
