import heapq
import math
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = float('inf')  # Cost from start node to current node
        self.h = float('inf')  # Heuristic cost from current node to goal node
        self.parent = None
    
    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)

def heuristic(node, goal):
    return math.sqrt((node.x - goal.x) ** 2 + (node.y - goal.y) ** 2)

def is_valid(x, y, obstacles):
    for ox, oy, diameter in obstacles:
        if (x - ox) ** 2 + (y - oy) ** 2 <= (diameter / 2) ** 2:
            return False
    return True

def get_neighbors(node, obstacles):
    neighbors = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
        nx, ny = node.x + dx, node.y + dy
        if 0 <= nx < 100 and 0 <= ny < 100 and is_valid(nx, ny, obstacles):
            neighbors.append(Node(nx, ny))
    return neighbors

def a_star(start, goal, obstacles):
    open_list = []
    closed_set = set()
    
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])
    
    start_node.g = 0
    start_node.h = heuristic(start_node, goal_node)
    
    heapq.heappush(open_list, start_node)
    
    while open_list:
        current_node = heapq.heappop(open_list)
        
        if current_node.x == goal_node.x and current_node.y == goal_node.y:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return list(reversed(path))
        
        closed_set.add((current_node.x, current_node.y))
        
        for neighbor in get_neighbors(current_node, obstacles):
            if (neighbor.x, neighbor.y) in closed_set:
                continue
            
            tentative_g = current_node.g + 1  # Assuming unit cost for each move
            
            if tentative_g < neighbor.g:
                neighbor.parent = current_node
                neighbor.g = tentative_g
                neighbor.h = heuristic(neighbor, goal_node)
                
                if neighbor not in open_list:
                    heapq.heappush(open_list, neighbor)
    
    return None  # No path found


def main():
    start = (10, 10)
    goal = (90, 90)
    obstacles = [(30, 30, 20), (50, 50, 20), (70, 10, 20)]

    path = a_star(start, goal, obstacles)
    if path:
        print("Path found:")
        print(path)
    else:
        print("Path not found")

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.title("A* Path Planning")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    # Plot grid
    for i in range(0, 101, 10):
        plt.plot([i, i], [0, 100], 'k-', alpha=0.2)
        plt.plot([0, 100], [i, i], 'k-', alpha=0.2)

    # Plot obstacles
    for ox, oy, diameter in obstacles:
        circle = plt.Circle((ox, oy), diameter / 2, color='gray')
        plt.gca().add_patch(circle)

    # Plot start and goal points
    plt.plot(start[0], start[1], 'ro', markersize=8, label="Start")
    plt.plot(goal[0], goal[1], 'go', markersize=8, label="Goal")

    # Plot path
    if path:
        path_x = [x for x, _ in path]
        path_y = [y for _, y in path]
        plt.plot(path_x, path_y, 'b-', linewidth=2, label="Path")

    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == "__main__":
    main()


