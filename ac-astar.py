import numpy as np
import time
import matplotlib.pyplot as plt
from queue import PriorityQueue

def initialize_grid(grid_size, obstacles, rect_obstacle):
    # Initialize grid
    grid = np.zeros((grid_size, grid_size))
    # Set circular obstacles in the grid
    for center, radius in obstacles:
        for x in range(grid_size):
            for y in range(grid_size):
                if (x - center[0])**2 + (y - center[1])**2 <= radius**2:
                    grid[x, y] = 1  # Mark as obstacle
    # Set rectangular obstacle
    x_coords, y_coords = zip(*rect_obstacle)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            grid[x, y] = 1  # Mark as obstacle
    return grid

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def a_star(start, goal, grid):
    queue = PriorityQueue()
    queue.put((0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not queue.empty():
        current = queue.get()[1]

        if current == goal:
            break

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Adjacent squares
            next = (current[0] + dx, current[1] + dy)
            if 0 <= next[0] < grid.shape[0] and 0 <= next[1] < grid.shape[1]:  # Grid boundaries
                if grid[next[0], next[1]] == 0:  # Check if not obstacle
                    new_cost = cost_so_far[current] + 1
                    if next not in cost_so_far or new_cost < cost_so_far[next]:
                        cost_so_far[next] = new_cost
                        priority = new_cost + heuristic(goal, next)
                        queue.put((priority, next))
                        came_from[next] = current
    return came_from if goal in came_from else None

def reconstruct_path(came_from, start, goal):
    if came_from is None:
        return None
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from.get(current)
    path.append(start)
    path.reverse()
    return path

def plot_path(grid, start, goal, path):
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.T, cmap='Greys', origin='lower', interpolation='none')
    plt.scatter(start[0], start[1], marker="o", color="green", s=100, label="Start")
    plt.scatter(goal[0], goal[1], marker="*", color="red", s=100, label="Goal")

    if path is not None:
        plt.plot(*zip(*path), marker=".", color="blue", label="Path")
        plt.title(f"Path Planning with A* Algorithm\nTotal steps: {len(path)}")
    else:
        plt.title("Path Planning with A* Algorithm\nNo valid path found!")

    plt.legend()
    plt.grid()
    plt.show()

def main():
    grid_size = 100
    start = (10, 10)
    goal = (90, 90)
    obstacles = [((30, 30), 10), ((50, 50), 10), ((70, 10), 10)]
    rect_obstacle = [(60, 80), (100, 80), (60, 70), (100, 70)]

    start_time = time.time()
    grid = initialize_grid(grid_size, obstacles, rect_obstacle)
    came_from = a_star(start, goal, grid)
    path = reconstruct_path(came_from, start, goal)
    end_time = time.time()

    print(f"Path planning took {end_time - start_time} seconds")

    plot_path(grid, start, goal, path)

if __name__ == "__main__":
    main()

