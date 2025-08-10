import json
import os
import pickle
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from rich import print


def curve_print(curve: list):
    if not curve:
        print("[]")
        return

    col_widths = [0] * 8
    for i, point in enumerate(curve):
        col_idx = i % 8
        s = str(point)
        if len(s) > col_widths[col_idx]:
            col_widths[col_idx] = len(s)

    total = len(curve)
    lines = total // 8
    remainder = total % 8

    print("[")
    for i in range(lines):
        start_idx = i * 8
        line_elements = []
        for j in range(8):
            element = curve[start_idx + j]
            s = str(element)
            line_elements.append(s.ljust(col_widths[j]))
        line_str = "    " + ", ".join(line_elements)
        if i < lines - 1 or remainder > 0:
            line_str += ","
        print(line_str)

    if remainder > 0:
        start_idx = lines * 8
        line_elements = []
        for j in range(remainder):
            element = curve[start_idx + j]
            s = str(element)
            line_elements.append(s.ljust(col_widths[j]))
        line_str = "    " + ", ".join(line_elements)
        print(line_str)
    print("]")


class _HilbertCurve:
    def __init__(
            self,
            order: int,
            info: bool
    ):
        """
        This function is a constructor for class `_HilbertCurve`
        Args:
            order (int): The order of Hilbert Curve's order.
        """
        self.order = order
        self.info = info

    def _build(self, order) -> tuple:
        """
        
        Args:
            order (int): The order of Hilbert Curve's order.

        Returns:
            tuple[
                list[
                    tuple[Any, Any]          or
                    tuple[Any, int or Any]   or
                    tuple[
                        int or Any,
                        int or Any
                    ]
                ], dict[
                    tuple[Any, Any]          or
                    tuple[Any, int or Any]   or
                    tuple[
                        int or Any,
                        int or Any
                    ],
                    int or Any
                ]
            ]

        """
        if order == 0:
            return [(0, 0)], {(0, 0): 0}
        # Generate order-1 order curves recursively
        lower_curve, lower_map = self._build(order - 1)
        self.size = 1 << (order - 1)  # 2^(order-1)
        curve = []
        coord_map = {}
        index = 0

        # The first quadrant: Rotation and flipping
        for x, y in lower_curve:
            new_x, new_y = y, x
            curve.append((new_x, new_y))
            coord_map[(new_x, new_y)] = index
            index += 1

        # The second quadrant: Translation
        for x, y in lower_curve:
            new_x, new_y = x, y + self.size
            curve.append((new_x, new_y))
            coord_map[(new_x, new_y)] = index
            index += 1

        # The third quadrant: Translation
        for x, y in lower_curve:
            new_x, new_y = x + self.size, y + self.size
            curve.append((new_x, new_y))
            coord_map[(new_x, new_y)] = index
            index += 1

        # The fourth quadrant: Rotation and flipping
        for x, y in lower_curve:
            new_x, new_y = 2 * self.size - 1 - y, self.size - 1 - x
            curve.append((new_x, new_y))
            coord_map[(new_x, new_y)] = index
            index += 1

        return curve, coord_map


class _HilbertMaze(_HilbertCurve):
    """"""
    def __init__(
            self,
            order: int,
            info: bool,
            save_dir: str = ".",
            save_name: str = "Hilbert_maze"
    ):
        """
        See parent class for more details.
        """
        super().__init__(order, info)
        self.save_dir = save_dir
        self.save_name = save_name
        self.curve = None
        self.maze  = None
        self.coord_map = None
        self.maze_start = self.maze_end = (-1, -1)
        self.solution_path = []  # 存储解路径坐标（按顺序）
        self.curve_maze_solve = []  # 存储解路径坐标（按坐标排序）

    # noinspection PyTypeChecker
    def _save_maze(self, path: str = None, filename: str = None):
        if path is None:
            path = self.save_dir
        if filename is None:
            filename = self.save_name + ".curve.txt"
        if self.maze is None:
            self._generate(self.order)
        with open(os.path.join(path, filename), "w", encoding="utf-8") as f:
            json.dump(self.curve, f)
        file2 = filename.split(".")[0] + ".coord_map"
        with open(os.path.join(path, file2), 'wb') as f:
            pickle.dump(self.coord_map, f)
        file3 = filename.split(".")[0] + ".maze"
        np.save(os.path.join(path, file3), self.maze)

    def _load_maze(self, maze_path, coord_map_path, curve_path):
        self.maze = np.load(maze_path)
        with open(curve_path, 'r', encoding="utf-8") as f:
            self.curve = json.load(f)
        with open(coord_map_path, 'rb') as f:
            self.coord_map = pickle.load(f)
        self.maze_start = self.curve[0]
        self.maze_end = self.curve[-1]

    def _generate(self, order: int) -> tuple[Any, Any, Any]:
        """
        Generate the maze using the given Hilbert curve.
        Args:
            order (int): The Hilbert Curve's order.

        Returns:
            tuple[Any, Any, Any]
        """
        curve, coord_map = self._build(order)
        size = 1 << order  # 2^order
        total_cells = size * size

        if self.info:
            print(f"[INFO] Curve Start : {curve[0]}")
            print(f"[INFO] Curve End   : {curve[-1]}")
            print(f"[INFO] Curve Length: {len(curve)}")
            print(f"[INFO] Full Curve  : \n>>>\n")
            curve_print(curve)
            print("\n<<<\n")
        self.curve = curve
        self.curve_start = curve[0]
        self.curve_end = curve[-1]
        self.coord_map = coord_map
        # The length of the list strictly follows the equation `f(x) = 4^x`

        # Create the maze grid (0= paths, 1= walls)
        # A `2*size+1` x `2*size+1` grid is used
        maze = np.ones((2 * size + 1, 2 * size + 1), dtype=int)

        # Mark all cell centers as pathways
        for i in range(size):
            for j in range(size):
                maze[2 * i + 1, 2 * j + 1] = 0

        # Save start and end coordinates (for entry/exit Settings)
        start_x, start_y = curve[0]
        end_x, end_y = curve[-1]

        # Connect adjacent cells (following Hilbert curve order)
        for idx, (x, y) in enumerate(curve):
            if idx == total_cells - 1:
                # The last cell has no further neighbors
                continue

            # Gets the position of the current cell in the maze grid
            forward_neighbors = []

            # Check the neighbors in all four directions
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    neighbor_idx = coord_map.get((nx, ny), -1)
                    # Consider only neighbors ahead of the curve (larger index)
                    if neighbor_idx > idx:
                        forward_neighbors.append((dx, dy, nx, ny))

            # Select a front neighbor at random and break through the wall
            if forward_neighbors:
                dx, dy, nx, ny = random.choice(forward_neighbors)
                # Calculate wall position (between current cell and neighbors)
                wall_x, wall_y = 2 * x + 1 + dx, 2 * y + 1 + dy
                maze[wall_x, wall_y] = 0  # Get through the wall

        # Break through the walls to set up entrances and exits
        # The exit is at the starting position (start of the curve)
        if start_y == 0:  # Left boundary
            maze[2 * start_x + 1, 0] = 0
        elif start_x == 0:  # Upper lateral boundary
            maze[0, 2 * start_y + 1] = 0

        # The entry is at the end (the end of the curve)
        if end_y == size - 1:  # Right boundary
            maze[2 * end_x + 1, 2 * size] = 0
        elif end_x == size - 1:  # Lower lateral boundary
            maze[2 * size, 2 * end_y + 1] = 0

        if self.info:
            print(f"[INFO] Maze Start: {curve[0]}")
            print(f"[INFO] Maze End  : {curve[-1]}")
            print(f"[INFO] Full Maze : \n>>>\n{maze}\n<<<\n")
        self.maze = maze
        self.maze_start = curve[0]
        self.maze_end = curve[-1]
        return self.maze, self.maze_start, self.maze_end

    def _show(self) -> None:
        """
        Visualize the maze.
        Returns:
            None
        """
        # Create a custom color mapping
        if self.maze is not None:
            maze = self.maze
            start = self.maze_start
            end = self.maze_end
        else:
            maze, start, end = self._generate(self.order)

        cmap = ListedColormap(['white', 'black', '#9F2B68', '#E49B0F'])

        # Copy the maze for visual marking
        display_maze = maze.copy().astype(float)

        # Mark the starting point (#E49B0F) and the ending point (#9F2B68)
        s_x, s_y = 2 * start[0] + 1, 2 * start[1] + 1
        e_x, e_y = 2 * end[0] + 1, 2 * end[1] + 1
        display_maze[s_x, s_y] = 2
        display_maze[e_x, e_y] = 3

        # Mark the entrance/exit path
        if start[1] == 0:  # Exit on the left
            display_maze[s_x, 0] = 2
        elif start[0] == 0:  # Upper exit
            display_maze[0, s_y] = 2

        if end[1] == (maze.shape[1] - 1) // 2 - 1:  # Entrance on the right
            display_maze[e_x, -1] = 3
        elif end[0] == (maze.shape[0] - 1) // 2 - 1:  # Lower entrance
            display_maze[-1, e_y] = 3

        plt.figure(figsize=(10, 10))
        plt.imshow(display_maze, cmap=cmap, interpolation='nearest')
        # Add grid lines for observation
        for i in range(0, maze.shape[0] + 1):
            plt.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        for j in range(0, maze.shape[1] + 1):
            plt.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.3)

        plt.title(f"Hilbert Maze\norder = {self.order}")
        plt.axis('off')
        plt.show()

    def _solve(self):
        """
        Solve the maze path using the DFS algorithm
        Returns: None

        """
        if self.maze is None:
            self._generate(self.order)

        # get maze's size
        size = 1 << self.order
        total_size = 2 * size + 1

        # the location of start and end in the maze
        start = (2 * self.maze_start[0] + 1, 2 * self.maze_start[1] + 1)
        end = (2 * self.maze_end[0] + 1, 2 * self.maze_end[1] + 1)

        # solve
        stack = [start]
        visited = np.zeros((total_size, total_size), dtype=bool)
        parent = {}
        visited[start] = True

        while stack:
            x, y = stack.pop()

            if (x, y) == end:
                path = []
                current = end
                while current != start:
                    path.append(current)
                    current = parent[current]
                path.append(start)
                path.reverse()
                self.solution_path = path
                curve_path = []
                for px, py in path:
                    if px % 2 == 1 and py % 2 == 1:
                        curve_path.append(((px - 1) // 2, (py - 1) // 2))
                self.curve_maze_solve = sorted(curve_path, key=lambda p: (p[0], p[1]))
                return

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < total_size and 0 <= ny < total_size and not visited[nx, ny]:
                    if self.maze[nx, ny] == 0 or (nx, ny) == end:
                        visited[nx, ny] = True
                        stack.append((nx, ny))
                        parent[(nx, ny)] = (x, y)

        raise RuntimeError("[Err ] Maze No Solution Error! This is should not occurred.")

    def _display_and_solve(self):
        if self.maze is None:
            self._generate(self.order)

        if not self.solution_path:
            self._solve()

        cmap = ListedColormap(['white', 'black', '#9F2B68', '#E49B0F', '#89cff0'])

        display_maze = self.maze.copy().astype(float)

        for (x, y) in self.solution_path:
            display_maze[x, y] = 4

        s_x, s_y = 2 * self.maze_start[0] + 1, 2 * self.maze_start[1] + 1
        e_x, e_y = 2 * self.maze_end[0] + 1, 2 * self.maze_end[1] + 1
        display_maze[s_x, s_y] = 2
        display_maze[e_x, e_y] = 3

        size = (display_maze.shape[1] - 1) // 2
        if self.maze_start[1] == 0:
            display_maze[s_x, 0] = 2
        elif self.maze_start[0] == 0:
            display_maze[0, s_y] = 2

        if self.maze_end[1] == size - 1:
            display_maze[e_x, 2 * size] = 3
        elif self.maze_end[0] == size - 1:
            display_maze[2 * size, e_y] = 3

        plt.figure(figsize=(10, 10))
        plt.imshow(display_maze, cmap=cmap, interpolation='nearest')

        for i in range(0, display_maze.shape[0] + 1):
            plt.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        for j in range(0, display_maze.shape[1] + 1):
            plt.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.3)

        plt.title(f"Hilbert Maze Solution\norder = {self.order}")
        plt.axis('off')
        plt.show()


class HilbertMaze(_HilbertMaze):
    VALID_INT_LENGTH = [8, 16, 32, 64, 128, 256]

    def __init__(
            self,
            order: int = 3,
            full_maze: bool = False,
            int_length: int = 16,
            info: bool = False,
            save_dir: str = ".",
            save_name: str = "Hilbert_maze"
    ):
        """
        See parent class for more details.
        Args:
            order      (int) : The order of Hilbert Curve's order
            full_maze  (bool): Whether to print the complete maze information
            int_length (int) : The integer type used when printing full maze information
        Returns: None
        Raises:
            ValueError :
                 - When `order` is not a value
                which can be converted to
                an integer greater than zero.

                 - When `int_length` is not a valid value(from [8, 16, 32, 64, 128, 256]).
        """

        try:
            order = int(order)
        except ValueError as e:
            raise ValueError(
                f"The `HilbertMaze.__init__`'s arg `order` " +
                f"must be a value that can be converted to an integer"
            ) from e
        if order <= 0:
            raise ValueError(
                f"The `HilbertMaze.__init__`'s arg `order` " +
                f"must greater than zero"
            )
        if int_length not in self.VALID_INT_LENGTH:
            raise ValueError(
                f"The `HilbertMaze.__init__`'s arg `int_length` " +
                f"can only be one of {self.VALID_INT_LENGTH}"
            )
        super().__init__(order, info, save_dir, save_name)
        self.full_maze = full_maze
        self.int_length = int_length
        if self.full_maze:
            np.set_printoptions(threshold=np.iinfo(getattr(np, f"int{self.int_length}")).max, linewidth=1000)

        self.maze = None


    def generate(self) -> None:
        self._generate(self.order)

    def display(self):
        self._show()

    def save_maze(self, **kwargs):
        self._save_maze(**kwargs)

    def load_maze(self, *args):
        self._load_maze(*args)

    def solve_and_display(self):
        self._display_and_solve()


if __name__ == "__main__":
    test = HilbertMaze(
        order=5,
        full_maze=True,
        info=False,
    )
    # test.load_maze("Hilbert_maze.maze.npy", "Hilbert_maze.coord_map", "Hilbert_maze.curve.txt")
    # test.display()
    test.solve_and_display()
    test.save_maze()
