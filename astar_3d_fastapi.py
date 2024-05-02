# A* path planning in a FastAPI implementation
#
import heapq
import numpy as np
from queue import PriorityQueue
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
import path_planning_utils as ppu
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.DEBUG, filename='sar_drone.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')

app = FastAPI()


class Location(BaseModel):
    latitude: float
    longitude: float
    height: float


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position  # a tuple (longitude, latitude, height)
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        # The hash of the Node is defined based on its position
        return hash(self.position)

    def __lt__(self, other):
        # This helps with sorting nodes if they're in a priority queue
        return self.f < other.f


class GeoGraph:
    def __init__(self, southwest_corner, northeast_corner, divisions):
        alt_min = 5  # default min altitude
        alt_max = 100  # default max
        lat_min, lon_min, alt_min = southwest_corner
        lat_max, lon_max, alt_max  = northeast_corner
        self.lats = np.linspace(lat_min, lat_max, divisions)
        self.lons = np.linspace(lon_min, lon_max, divisions)
        self.alts = np.linspace(alt_min, alt_max, divisions)
        self.nodes = [(lat, lon, alt) for lat in self.lats for lon in self.lons for alt in self.alts]
        self.edges = {node: [] for node in self.nodes}
        self.connect_nodes()

    def connect_nodes(self):
        lat_steps = len(self.lats)
        lon_steps = len(self.lons)
        alt_steps = len(self.alts)
        for i, (lat, lon, alt) in enumerate(self.nodes):
            if i % lon_steps != 0:  # Not on the western edge
                self.edges[(lat, lon, alt)].append(self.nodes[i - 1])  # Connect west
            if i % lon_steps != lon_steps - 1:  # Not on the eastern edge
                self.edges[(lat, lon, alt)].append(self.nodes[i + 1])  # Connect east
            if i >= lon_steps:  # Not on the northern edge
                self.edges[(lat, lon, alt)].append(self.nodes[i - lon_steps])  # Connect north
            if i < (lat_steps * lon_steps - lon_steps):  # Not on the southern edge
                self.edges[(lat, lon, alt)].append(self.nodes[i + lon_steps])  # Connect south

    def neighbors(self, node):
        return self.edges[node]

    def cost(self, a, b):
        return ppu.llh_to_meters(a, b)[1]


# The A* heuristic function
def heuristic(node, goal):
    return ppu.llh_to_meters(node, goal)[1]


# The classic way to do A*
def a_star(start, goal, graph):
    queue = PriorityQueue()
    queue.put((0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not queue.empty():
        current = queue.get()[1]

        if current == goal:
            came_from[goal] = current

            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal)
                queue.put((priority, next))
                came_from[next] = current

    logging.debug(f"nodes: {came_from}\n")
    logging.debug(f"start: {start}\n")
    logging.debug(f"goal: {goal}\n")
    logging.debug(f"\nnodes length: {len(came_from)}")

    return came_from


# A new (unproven?) way to do A*
def astar_3d_geo(start, end):
    open_list = []
    closed_list = set()

    start_node = Node(None, start)
    end_node = Node(None, end)

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node)

        if current_node == end_node:
            return return_path(current_node)

        neighbors = generate_neighbors(current_node.position)

        for next in neighbors:
            neighbor = Node(current_node, next)
            if neighbor in closed_list:
                continue

            neighbor.g = current_node.g + ppu.hhl_to_meters(current_node.position, neighbor.position)
            neighbor.h = ppu.hhl_to_meters(neighbor.position, end_node.position)
            neighbor.f = neighbor.g + neighbor.h

            if add_to_open(open_list, neighbor):
                heapq.heappush(open_list, neighbor)
    return None


def return_path(nodes, start, goal):
    path = []
    current_node = nodes[goal]
    while current_node.position != start and current_node is not None:
        path.append(current_node.position)
        current_node = current_node.parent
    path.append(start)
    path.reverse()
    return path


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


def add_to_open(open_list, neighbor):
    for node in open_list:
        if neighbor == node and neighbor.g > node.g:
            return False
    return True


def generate_neighbors(position):
    step_deg = 0.00001  # step size in degrees, adjustable
    step_height = 1  # Step size in meters for altitude changes

    (lon, lat, height) = position
    neighbors = []

    # 8 surrounding directions + stay in place, with three altitude changes
    for dlon in [-step_deg, 0, step_deg]:
        for dlat in [-step_deg, 0, step_deg]:
            for dheight in [-step_height, 0, step_height]:
                if dlon == 0 and dlat == 0 and dheight == 0:
                    continue  # Skip the stay in place position
                new_position = (lon + dlon, lat + dlat, height + dheight)
                neighbors.append(new_position)

    return neighbors


@app.post("/astar/")
async def astar_route(start: Tuple[float, float, float], goal: Tuple[float, float, float], obstacles: List[Tuple[Tuple[float, float, float], float]]):
    # Create grid graph
    divisions = 100
    graph = GeoGraph(start, goal, divisions)
    nodes = a_star(start, goal, graph)
    logging.debug(f"\nnodes: {len(nodes)}")
    path = return_path(nodes, start, goal)
    #path = astar_3d_geo(start, goal)

    mission_plan = ppu.path_to_mission(path)

    return mission_plan
