import numpy as np
import matplotlib.pyplot as plt
import random
import time
from math import sqrt, inf

def is_collision(point, obstacles, rect_obstacle):
    for center, radius in obstacles:
        if sqrt((center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2) <= radius:
            return True
    x, y = point
    if rect_obstacle[0][0] <= x <= rect_obstacle[3][0] and rect_obstacle[0][1] <= y <= rect_obstacle[1][1]:
        return True
    return False

def nearest_node(nodes, point):
    nearest = None
    min_dist = inf
    for node in nodes:
        dist = sqrt((node[0] - point[0]) ** 2 + (node[1] - point[1]) ** 2)
        if dist < min_dist:
            nearest = node
            min_dist = dist
    return nearest

def steer(from_node, to_point, step_size, obstacles, rect_obstacle):
    direction = np.arctan2(to_point[1] - from_node[1], to_point[0] - from_node[0])
    new_point = (from_node[0] + step_size * np.cos(direction), from_node[1] + step_size * np.sin(direction))
    return new_point if not is_collision(new_point, obstacles, rect_obstacle) else None

def connect(start_tree, goal_tree, step_size, obstacles, rect_obstacle):
    last_node = None
    while True:
        nearest = nearest_node(start_tree, last_node if last_node else goal_tree[-1])
        new_point = steer(nearest, goal_tree[-1], step_size, obstacles, rect_obstacle)
        if new_point is None:
            return last_node  # Can't extend further towards the goal_tree
        start_tree[new_point] = nearest
        last_node = new_point
        if new_point == goal_tree[-1]:
            return new_point  # Trees have connected

def build_rrt_connect(start, goal, grid_size, step_size, max_iterations, obstacles, rect_obstacle):
    start_tree, goal_tree = {start: None}, {goal: None}
    for i in range(max_iterations):
        if i % 2 == 0:  # Alternate between expanding the start_tree and goal_tree
            random_point = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
            if is_collision(random_point, obstacles, rect_obstacle):
                continue
            nearest = nearest_node(start_tree.keys(), random_point)
            new_point = steer(nearest, random_point, step_size, obstacles, rect_obstacle)
            if new_point:
                start_tree[new_point] = nearest
                if connect(start_tree, goal_tree, step_size, obstacles, rect_obstacle) == goal:
                    return start_tree, goal_tree
        else:
            random_point = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
            if is_collision(random_point, obstacles, rect_obstacle):
                continue
            nearest = nearest_node(goal_tree.keys(), random_point)
            new_point = steer(nearest, random_point, step_size, obstacles, rect_obstacle)
            if new_point:
                goal_tree[new_point] = nearest
                if connect(goal_tree, start_tree, step_size, obstacles, rect_obstacle) == start:
                    return start_tree, goal_tree
    return start_tree, goal_tree  # May return incomplete trees if no path found

def reconstruct_path(trees, start, goal):
    # Combine paths from start to meeting point and from goal to meeting point
    meeting_point = trees[1][goal]
    path = [meeting_point]
    while path[-1] != start:
        path.append(trees[0][path[-1]])
    path.reverse()
    current = meeting_point
    while current != goal:
        current = trees[1][current]
        path.append(current)
    return path

def main():
    grid_size = 100
    start = (10, 10)
    goal = (90, 90)
    step_size = 2  # Smaller step size for more detailed path
    max_iterations = 10000
    obstacles = [((30, 30), 10), ((50, 50), 10), ((70, 10), 10)]
    rect_obstacle = [(60, 70), (100, 70), (100, 80), (60, 80)]

    start_time = time.time()
    start_tree, goal_tree = build_rrt_connect(start, goal, grid_size, step_size, max_iterations, obstacles, rect_obstacle)
    path = reconstruct_path((start_tree, goal_tree), start, goal)
    end_time = time.time()

    print(f"Path planning took {end_time - start_time} seconds")

    plt.figure(figsize=(12, 12))
    for point, parent in start_tree.items():
        if parent:
            plt.plot([point[0], parent[0]], [point[1], parent[1]], 'r-')
    for point, parent in goal_tree.items():
        if parent:
            plt.plot([point[0], parent[0]], [point[1], parent[1]], 'b-')
    for obstacle in obstacles:
        circle = plt.Circle(obstacle[0], obstacle[1], radius=obstacle[1], color='k', fill=True)
        plt.gca().add_patch(circle)
    rect = plt.Rectangle(rect_obstacle[0], rect_obstacle[1][0] - rect_obstacle[0][0], rect_obstacle[2][1] - rect_obstacle[0][1], color='k', fill=True)
    plt.gca().add_patch(rect)
    plt.plot(*zip(*path), marker='o', color='g', markersize=5, linestyle='-', linewidth=2, label='RRT-Connect Path')
    plt.plot(start[0], start[1], 'go', markersize=10, label="Start")
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label="Goal")
    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)
    plt.title('RRT-Connect Path Planning\nTotal steps: {len(path)}')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

