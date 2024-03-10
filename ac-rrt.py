import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt
import time

def create_obstacle_patch(obstacle):
    if len(obstacle) == 2:  # Circular obstacle
        center, radius = obstacle
        return plt.Circle(center, radius, color='k', fill=True)
    elif len(obstacle) == 4:  # Rectangular obstacle
        bottom_left, _, top_right, _ = sorted(obstacle)  # Assuming format: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        width, height = top_right[0] - bottom_left[0], top_right[1] - bottom_left[1]
        return plt.Rectangle(bottom_left, width, height, color='k', fill=True)
    return None

def is_collision(point, obstacles, rect_obstacle):
    for center, radius in obstacles:
        if sqrt((center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2) <= radius:
            return True
    x, y = point
    if rect_obstacle[0][0] <= x <= rect_obstacle[2][0] and rect_obstacle[0][1] <= y <= rect_obstacle[2][1]:
        return True
    return False

def nearest_node(nodes, point):
    return min(nodes, key=lambda x: sqrt((x[0] - point[0]) ** 2 + (x[1] - point[1]) ** 2))

def steer(from_node, to_point, step_size, obstacles, rect_obstacle):
    direction = np.arctan2(to_point[1] - from_node[1], to_point[0] - from_node[0])
    new_point = (from_node[0] + step_size * np.cos(direction), from_node[1] + step_size * np.sin(direction))
    return new_point if not is_collision(new_point, obstacles, rect_obstacle) else None

def build_rrt(start, goal, grid_size, step_size, max_iterations, obstacles, rect_obstacle):
    nodes = {start: None}
    for _ in range(max_iterations):
        random_point = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
        nearest = nearest_node(nodes.keys(), random_point)
        new_point = steer(nearest, random_point, step_size, obstacles, rect_obstacle)
        if new_point:
            nodes[new_point] = nearest
            if sqrt((new_point[0] - goal[0]) ** 2 + (new_point[1] - goal[1]) ** 2) <= step_size:
                nodes[goal] = new_point
                break
    return nodes

def reconstruct_path(nodes, start, goal):
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = nodes.get(current)
    path.reverse()
    return path

def main():
    grid_size = 100
    start = (10, 10)
    goal = (90, 90)
    step_size = 5
    max_iterations = 5000
    obstacles = [((30, 30), 10), ((50, 50), 10), ((70, 10), 10)]
    rect_obstacle = [(60, 70), (100, 70), (100, 80), (60, 80)]

    start_time = time.time()
    nodes = build_rrt(start, goal, grid_size, step_size, max_iterations, obstacles, rect_obstacle)
    path = reconstruct_path(nodes, start, goal)
    end_time = time.time()

    print(f"Path planning took {end_time - start_time} seconds")

    plt.figure(figsize=(12, 12))
    for obstacle in obstacles + [rect_obstacle]:
        patch = create_obstacle_patch(obstacle)
        if patch:
            plt.gca().add_patch(patch)
    for (node, parent) in nodes.items():
        if parent:
            plt.plot([node[0], parent[0]], [node[1], parent[1]], 'gray', linewidth=0.5)  # Intermediary steps
    if path:
        plt.plot(*zip(*path), marker='o', color='blue', markersize=5, linestyle='-', linewidth=2, label="Path")
    plt.plot(start[0], start[1], 'go', markersize=10, label="Start")
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label="Goal")
    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)
    plt.title(f'RRT Path Planning - Steps: {len(path)}')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()

