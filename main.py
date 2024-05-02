import requests
import sys
import logging
import pymavlink_SITL as sitl

# Set up basic configuration for logging
logging.basicConfig(level=logging.DEBUG, filename='sar_drone.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


# Define the base URL
base_url = 'http://127.0.0.1'

# Defaul API server ports
DQN_port = ':5000'
AStar_port = ':5010'
RRT_port = ':5020'


# Flask API preparation
def set_start(start: tuple, goal: tuple) -> None:
    """
    Set the start and goal positions for the mission.

    Sends a request to the server API to assign the start and goal
    positions for the mission. Prints a success message if the request is
    successful; otherwise, prints an error message.

    Parameters:
    - start (tuple): Tuple containing the start position coordinates
                     (lat, lon, alt).
    - goal (tuple): Tuple containing the goal position coordinates
                    (lat, lon, alt).
    """
    # Construct the URL with start and goal parameters
    extension = '/assign-start-goal?start={}_{}_{}&goal={}_{}_{}'
    url = base_url + DQN_port + extension.format(*start, *goal)
    # Send a GET request to the server API
    response = requests.get(url)
    # Check the response status code
    if response.status_code == 200:
        logging.info("Set start successfully!")
    else:
        logging.info("Error while tryig to start!")


# Request path from DQN server API
def get_path_from_api() -> str:
    """
    Retrieve the path information from the API endpoint.

    Makes a GET request to the server API to retrieve the path information.
    Returns the path content if the request is successful; otherwise, prints
    an error message and returns None.

    Returns:
    - str: Path content retrieved from the API.
    """
    # Make a GET request to the API endpoint
    response = requests.get(base_url + DQN_port + '/get-path')
    # Check if the request was successful (status code 200)
    if response.status_code == 201:
        # Extract the path content from the response
        path_content = response.text
        return path_content
    else:
        # Handle any errors or unexpected status codes
        logging.error('DQN Path Request Failed! Error Code:', response.status_code)
        return None


# Call the DQN Flask API
def call_DQN_API(start, goal):
    # Call the DQN
    set_start(start, goal)
    # Call the function to get path content from the API
    path_content = get_path_from_api()
    # Check if path content was retrieved successfully
    if path_content is not None:
        return path_content
    else:
        logging.error('Failed to retrieve DQN path.')
        return None


# Call the A* FastAPI
def call_astar_endpoint(start, goal, obstacles):
    url = base_url + AStar_port + '/astar/'
#    url = 'http://127.0.0.1:5010/astar/'
    payload = {
        'start': start,
        'goal': goal,
        'obstacles': obstacles,
    }
    logging.info(f"A* API: {url}")
    logging.info(f"A* Request: {payload}")

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        logging.info("A* mission received OK")
        return response.json()
    else:
        logging.error("A* Route Request Failed! Status Code:", response.status_code)

def call_rrt_endpoint(start, goal, obstacles):
    url = base_url + RRT_port + '/rrt/'
#    url = 'http://127.0.0.1:5020/rrt/'
    payload = {
        'start': start,
        'goal': goal,
        'obstacles': obstacles
    }
    logging.info(f"RRT API: {url}")
    logging.info(f"RRT Request: {payload}")

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        logging.info("RRT mission plan received OK")
        return response.json()
    else:
        logging.error("RRT Route Request Failed! Status Code:", response.status_code)


def main():
    """
    Main function get path from DQN Network, A* algorithm and RRT algorithm

    Retrieves start and goal coordinates from command-line arguments or uses
    default values. It then calls API to get mission plans and execute it.
    """
    # Check if command-line arguments are provided
    if len(sys.argv) == 1:
        # Default start and goal coordinates
        start = (42.355118, -71.071305, 6)
        goal = (42.355280, -71.070911, 26.5)
        obstacles = [((42.355200, -71.071000, 10.0), 5), ((42.355150, -71.071200, 5.0), 2)]
    else:
        # Check if all required arguments are provided
        if len(sys.argv) < 7:
            print('Usage:')
            print('python demo_api_test.py')
            print('or')
            print('python demo_api_test.py s1 s2 s3 g1 g2 g3')
            exit(1)
        # Parse start and goal coordinates from command-line arguments
        start = sys.argv[1:4]
        goal = sys.argv[4:7]

    # Request DQN mission plan
    dqn_mission_plan = call_DQN_API(start, goal)
    print(dqn_mission_plan)

    # Request A* mission plan
    astar_mission_plan = call_astar_endpoint(start, goal, obstacles)
    print(astar_mission_plan)

    # Requet RRT mission plan
    rrt_mission_plan = call_rrt_endpoint(start, goal, obstacles)
    for wp in rrt_mission_plan:
        print(wp)


if __name__ == '__main__':
    main()
