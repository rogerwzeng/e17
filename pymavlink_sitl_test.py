import argparse
import time
from pymavlink import mavutil


# Connect to the Vehicle
def connect_vehicle(connection_string):

    if not connection_string :
        connection_string = 'udp:127.0.0.1:14551'  # default link port

    print('Connecting to vehicle on: %s' % connection_string)

    # Connect to SITL output using the appropriate IP and port
    vehicle = mavutil.mavlink_connection(connection_string)
    vehicle.wait_heartbeat()
    print(f"Heartbeat from system {vehicle.target_system} component {vehicle.target_component}")

    return vehicle

# Function to arm and take off
def arm_and_takeoff(vehicle, target_altitude):
    guided_mode = vehicle.mode_mapping()["GUIDED"]
    #print(f"Guided Mode is {guided_mode}")
    takeoff_params = [0,0,0,0,0,0,target_altitude]

    vehicle.mav.command_long_send(
        vehicle.target_system,
        vehicle.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        guided_mode, 0, 0, 0, 0, 0)
    ack_msg = vehicle.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
    print(f"Change mode to Guided: {ack_msg}")

    # Confirm vehicle armed before attempting to take off
    print("Arming motors")
    vehicle.mav.command_long_send(
        vehicle.target_system,
        vehicle.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0)  # 1 to arm

    # Wait for arming
    vehicle.motors_armed_wait()
    print("Motors armed!")

    print("Taking off!")
    vehicle.mav.command_long_send(
        vehicle.target_system,
        vehicle.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, target_altitude)

    # Wait until the vehicle reaches a safe height
    while True:
        # Request altitude
        vehicle.mav.request_data_stream_send(vehicle.target_system, vehicle.target_component, mavutil.mavlink.MAV_DATA_STREAM_POSITION, 1, 1)
        msg = vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        altitude = msg.alt / 1000.0  # Convert from mm to m

        print("Altitude:", altitude)
        if altitude >= target_altitude * 0.95:  # Just below target, in meters
            print("Reached target altitude")
            break
        time.sleep(1)

# Landing
def land_vehicle(connection):
    print("Landing")
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0)

    print("Landing command sent")

# Clear any existing mission
def clear_mission(connection):
    connection.mav.mission_clear_all_send(connection.target_system, connection.target_component)
    while True:
        msg = connection.recv_match(type='MISSION_ACK', blocking=True)
        if msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
            print("Mission cleared")
            break

def upload_mission(connection, filename):
    with open(filename, 'r') as file:
        first_line = file.readline()
        if not first_line.startswith('QGC WPL 110'):
            raise Exception('File format not supported')

        mission_items = []
        for line in file:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split('\t')
            lat, lon, alt = map(float, parts[8:11])
            seq = int(parts[0])
            frame = int(parts[2])
            command = int(parts[3])
            mission_items.append(mavutil.mavlink.MAVLink_mission_item_message(
                connection.target_system, connection.target_component, seq, frame,
                command, 0, 0, 0, 0, 0, 0, lat, lon, alt))

        connection.mav.mission_count_send(connection.target_system, connection.target_component, len(mission_items))

        for i, item in enumerate(mission_items):
            msg = connection.recv_match(type='MISSION_REQUEST', blocking=True)
            connection.mav.send(item)

        msg = connection.recv_match(type='MISSION_ACK', blocking=True)
        if msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
            print("Mission upload SUCCESS!")
        else:
            print("Mission upload FAILED!")
        
        return len(mission_items)

# Set the drone to ARMED AUTO mode
def set_mode_auto(vehicle):
    # Change mode to AUTO
    auto_mode = vehicle.mode_mapping()["AUTO"]
    print(f"Auto Mode is {auto_mode}")

    vehicle.mav.command_long_send(
        vehicle.target_system,
        vehicle.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        auto_mode, 0, 0, 0, 0, 0)
    ack_msg = vehicle.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)

    print("Mode set to AUTO")


# Protected main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='commands')
    parser.add_argument('--connect')
    parser.add_argument('--mission')
    args = parser.parse_args()

    # Connect to quadcopter
    connection_string = args.connect
    mission_file = args.mission

    print(f"Connection String: {connection_string} ; Mission: {mission_file}")

    # Mission workflow
    copter = connect_vehicle(connection_string)
    clear_mission(copter)
    waypoints = upload_mission(copter, mission_file)
    arm_and_takeoff(copter,20)  # initial height 20m
    time.sleep(5)  # let the copter catches its breath, before ..
    set_mode_auto(copter)  # execute the mission

    # Monitor mission progress
    current_wp = 0
    while True:
        next_wp = copter.recv_match(type='MISSION_CURRENT', blocking=True, timeout=5).seq
        if next_wp == current_wp:
            continue

        # We've reached a new waypoint
        print(f"Reached waypoint {next_wp}")
        current_wp = next_wp

        # Have we done all waypoints?
        if next_wp == waypoints - 1:
            # Land and finish
            land_vehicle(copter)
            print("Mission Completed")
            break
        time.sleep(5)


